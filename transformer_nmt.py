import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==========================================
# 1. 归一化层：LayerNorm vs RMSNorm
# ==========================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))

def get_norm_layer(d_model, norm_type='layernorm'):
    """获取归一化层"""
    if norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

# ==========================================
# 2. 位置编码：绝对 vs 相对
# ==========================================

class AbsolutePositionalEncoding(nn.Module):
    """绝对位置编码（原始Transformer）"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    """相对位置编码（简化版，基于学习的位置嵌入）"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # 学习相对位置嵌入
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, d_model) * 0.02)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # 计算相对位置索引
        positions = torch.arange(seq_len, device=x.device)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        rel_positions = rel_positions + self.max_len - 1  # 转换为非负索引
        
        # 限制在有效范围内
        rel_positions = torch.clamp(rel_positions, 0, 2 * self.max_len - 2)
        
        # 获取相对位置嵌入并加到输入
        rel_emb = self.rel_pos_emb[rel_positions]  # [seq_len, seq_len, d_model]
        # 对每个位置，使用其相对位置嵌入的平均值（简化处理）
        rel_emb = rel_emb.mean(dim=1)  # [seq_len, d_model]
        x = x + rel_emb.unsqueeze(0)
        return self.dropout(x)

def get_pos_encoding(d_model, max_len=5000, pos_type='absolute', dropout=0.1):
    """获取位置编码"""
    if pos_type == 'absolute':
        return AbsolutePositionalEncoding(d_model, max_len, dropout)
    elif pos_type == 'relative':
        return RelativePositionalEncoding(d_model, max_len, dropout)
    else:
        raise ValueError(f"Unknown pos_type: {pos_type}")

# ==========================================
# 3. Multi-Head Attention
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output

# ==========================================
# 4. Feed Forward Network
# ==========================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ==========================================
# 5. Encoder Layer
# ==========================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(d_model, norm_type)
        self.norm2 = get_norm_layer(d_model, norm_type)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# ==========================================
# 6. Decoder Layer
# ==========================================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(d_model, norm_type)
        self.norm2 = get_norm_layer(d_model, norm_type)
        self.norm3 = get_norm_layer(d_model, norm_type)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# ==========================================
# 7. Transformer Encoder
# ==========================================

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_len=5000, dropout=0.1, pos_type='absolute', norm_type='layernorm'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = get_pos_encoding(d_model, max_len, pos_type, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # x: [batch, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# ==========================================
# 8. Transformer Decoder
# ==========================================

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_len=5000, dropout=0.1, pos_type='absolute', norm_type='layernorm'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = get_pos_encoding(d_model, max_len, pos_type, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: [batch, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(x)
        return output

# ==========================================
# 9. Transformer Seq2Seq Model
# ==========================================

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 n_layers=6, d_ff=2048, max_len=5000, dropout=0.1,
                 pos_type='absolute', norm_type='layernorm', device='cuda'):
        super().__init__()
        self.device = device
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff,
            max_len, dropout, pos_type, norm_type
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff,
            max_len, dropout, pos_type, norm_type
        )
    
    def make_src_mask(self, src):
        # src: [batch, src_len]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        # tgt: [batch, tgt_len]
        tgt_len = tgt.shape[1]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
        # 添加因果掩码（防止看到未来信息）
        nopeak_mask = torch.triu(torch.ones((1, tgt_len, tgt_len), device=self.device), diagonal=1) == 0
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask.to(self.device)
    
    def forward(self, src, tgt):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
    
    def translate_beam(self, src, beam_width=5, max_len=50, sos_idx=1, eos_idx=2):
        """Beam Search解码"""
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_output = self.encoder(src, src_mask)
            
            # 初始化beam
            beams = [(0, [sos_idx])]
            
            for _ in range(max_len):
                new_beams = []
                for score, seq in beams:
                    if seq[-1] == eos_idx:
                        new_beams.append((score, seq))
                        continue
                    
                    tgt = torch.tensor([seq]).to(self.device)
                    tgt_mask = self.make_tgt_mask(tgt)
                    output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
                    
                    # 获取最后一个位置的log概率
                    log_probs = F.log_softmax(output[0, -1, :], dim=0)
                    top_probs, top_idxs = log_probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        new_score = score + top_probs[i].item()
                        new_seq = seq + [top_idxs[i].item()]
                        new_beams.append((new_score, new_seq))
                
                # 保留top-k beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                
                # 如果所有beam都结束，提前退出
                if all(b[1][-1] == eos_idx for b in beams):
                    break
            
            return beams[0][1]

    def translate_greedy(self, src, max_len=50, sos_idx=1, eos_idx=2):
        """贪婪解码（快速版本）"""
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_output = self.encoder(src, src_mask)

            # 初始化
            seq = [sos_idx]

            for _ in range(max_len):
                if seq[-1] == eos_idx:
                    break

                tgt = torch.tensor([seq]).to(self.device)
                tgt_mask = self.make_tgt_mask(tgt)
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

                # 获取最后一个位置的log概率
                log_probs = F.log_softmax(output[0, -1, :], dim=0)

                # 贪婪选择概率最大的词
                next_token = log_probs.argmax().item()
                seq.append(next_token)

            return seq

    def translate(self, src, method='greedy', beam_width=5, max_len=50, sos_idx=1, eos_idx=2):
        """统一解码接口"""
        if method == 'greedy':
            return self.translate_greedy(src, max_len=max_len, sos_idx=sos_idx, eos_idx=eos_idx)
        elif method == 'beam':
            return self.translate_beam(src, beam_width=beam_width, max_len=max_len, sos_idx=sos_idx, eos_idx=eos_idx)
        else:
            raise ValueError(f"Unknown decode method: {method}. Choose 'greedy' or 'beam'.")

