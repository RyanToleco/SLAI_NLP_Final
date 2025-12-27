import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ==========================================
# 1. Attention 模块 (支持三种对齐函数)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hid_dim, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            # Multiplicative (乘法): h_t^T * W * h_s
            self.wa = nn.Linear(hid_dim, hid_dim, bias=False)
        elif method == 'concat':
            # Additive (加法): v^T * tanh(W * [h_t; h_s])
            self.wa = nn.Linear(hid_dim * 2, hid_dim, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hid_dim))
            nn.init.uniform_(self.va, -0.1, 0.1)
            
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim] (Decoder 上一层最后一层的隐藏状态)
        # encoder_outputs: [batch, src_len, hid_dim]
        batch_size, src_len, _ = encoder_outputs.shape
        
        if self.method == 'dot':
            # 点积: [B, 1, H] * [B, H, S] -> [B, 1, S]
            attn_energies = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2))
        elif self.method == 'general':
            transformed = self.wa(encoder_outputs)
            attn_energies = torch.bmm(hidden.unsqueeze(1), transformed.transpose(1, 2))
        elif self.method == 'concat':
            h_rep = hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.wa(torch.cat((h_rep, encoder_outputs), dim=2)))
            # [B, S, H] * [H] -> [B, S] -> [B, 1, S]
            attn_energies = torch.matmul(energy, self.va).unsqueeze(1)
            
        return F.softmax(attn_energies.squeeze(1), dim=1)

# ==========================================
# 2. Encoder 模块 (两层单向 GRU)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 明确：两层单向 GRU
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch, src_len, hid_dim]
        # hidden: [2, batch, hid_dim]
        return outputs, hidden

# ==========================================
# 3. Decoder 模块 (带 Attention)
# ==========================================
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=2, dropout=0.5, attn_method='dot'):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(hid_dim, attn_method)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # GRU 输入是：当前词 Embedding + Attention 计算出的 Context Vector
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch]
        # hidden: [2, batch, hid_dim]
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        
        # 使用 Decoder 最后一层的隐藏状态计算 Attention
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        # context: [batch, 1, hid_dim]
        context = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 最终预测结合了 RNN 输出和上下文信息
        prediction = self.fc_out(torch.cat((output, context), dim=2).squeeze(1))
        return prediction, hidden

# ==========================================
# 4. Seq2Seq 封装 (支持 TF 策略与 Beam Search)
# ==========================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, tf_ratio=0.5):
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        enc_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_outputs)
            outputs[t] = output
            # Teacher Forcing 策略对比
            teacher_force = random.random() < tf_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

    def translate_beam(self, src, beam_width=3, max_len=50, sos_idx=1, eos_idx=2):
        """支持 Greedy (beam_width=1) 和 Beam Search"""
        self.eval()
        with torch.no_grad():
            enc_outputs, hidden = self.encoder(src)
            # (累计 log 概率, 当前序列, 隐藏状态)
            beams = [(0, [sos_idx], hidden)]
            
            for _ in range(max_len):
                new_beams = []
                for score, seq, h in beams:
                    if seq[-1] == eos_idx:
                        new_beams.append((score, seq, h))
                        continue
                    
                    input_tensor = torch.tensor([seq[-1]]).to(self.device)
                    output, next_h = self.decoder(input_tensor, h, enc_outputs)
                    log_probs = F.log_softmax(output, dim=1)
                    
                    top_probs, top_idxs = log_probs.topk(beam_width)
                    for i in range(beam_width):
                        new_beams.append((score + top_probs[0][i].item(), seq + [top_idxs[0][i].item()], next_h))
                
                # 排序并保留前 k 个分支
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                if all(b[1][-1] == eos_idx for b in beams):
                    break
                    
        return beams[0][1]