import json
import re
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import jieba

# --- 1. Data Cleaning & Tokenization ---
def clean_text(text):
    """移除非法字符，保留中英文、数字及基本标点"""
    text = re.sub(r'[^\w\s\u4e00-\u9fa5\u3000-\u303f\uff01-\uff5e.,!?;:\"\'\-]', '', text)
    return text.strip()

def tokenize_zh(text):
    """中文分词 (Jieba)"""
    return list(jieba.cut(text))

def tokenize_en(text):
    """英文分词 (类似 NLTK 处理方式)"""
    text = text.lower()
    # 分离单词与标点
    return re.findall(r'\w+|[^\w\s]', text)

# --- 2. Vocabulary Construction ---
def build_vocab(tokenized_sentences, min_freq=2, specials=['<PAD>', '<SOS>', '<EOS>', '<UNK>']):
    """构建统计词表，过滤低频词"""
    counter = collections.Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)
    
    vocab = {spec: i for i, spec in enumerate(specials)}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    
    itos = {i: token for token, i in vocab.items()}
    return vocab, itos

# --- 3. Word Embedding Initialization ---
def load_pretrained_embeddings(vocab, embedding_file, embed_dim):
    """
    根据词表加载预训练词向量
    embedding_file: 预训练文件路径 (如 glove.6B.100d.txt)
    """
    # 初始化为随机正态分布
    embeddings = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    embeddings[vocab['<PAD>']] = np.zeros((embed_dim,))
    
    if embedding_file and os.path.exists(embedding_file):
        print(f"Loading pretrained embeddings from {embedding_file}...")
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab:
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[vocab[word]] = vector
                    
    return torch.FloatTensor(embeddings)

# --- 4. Dataset Processing ---
def process_file(file_path, max_len=50):
    """读取、清洗、分词并过滤长难句

    注意：序列在Dataset中会添加<SOS>和<EOS>标记，
    所以这里需要限制为 max_len-2，以确保最终长度不超过max_len
    """
    src_data, tgt_data = [], []
    actual_max_len = max_len - 2  # 预留<SOS>和<EOS>标记的位置
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            zh = clean_text(item['zh'])
            en = clean_text(item['en'])
            
            zh_tokens = tokenize_zh(zh)
            en_tokens = tokenize_en(en)
            
            # 过滤过长或为空的句子
            if 0 < len(zh_tokens) <= actual_max_len and 0 < len(en_tokens) <= actual_max_len:
                src_data.append(zh_tokens)
                tgt_data.append(en_tokens)
    return src_data, tgt_data

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_tokens, tgt_tokens, src_vocab, tgt_vocab):
        self.src_data = [[src_vocab.get(tok, src_vocab['<UNK>']) for tok in sent] for sent in src_tokens]
        self.tgt_data = [[tgt_vocab.get(tok, tgt_vocab['<UNK>']) for tok in sent] for sent in tgt_tokens]
        
    def __len__(self): return len(self.src_data)
        
    def __getitem__(self, idx):
        # 加上开始和结束符
        src = [1] + self.src_data[idx] + [2]
        tgt = [1] + self.tgt_data[idx] + [2]
        return src, tgt

def collate_fn(batch):
    """用于 DataLoader 的补齐逻辑"""
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_list.append(torch.tensor(src))
        tgt_list.append(torch.tensor(tgt))
    
    from torch.nn.utils.rnn import pad_sequence
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return src_padded, tgt_padded