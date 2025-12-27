import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import random
import numpy as np
import argparse
import os
import json
import csv
import logging
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, modified_precision

# 导入自定义模块
from preprocess import process_file, build_vocab, TranslationDataset, collate_fn
from rnn_nmt import Encoder, Decoder, Seq2Seq
from transformer_nmt import TransformerNMT

# ==========================================
# 0. 日志和输出目录设置
# ==========================================

def setup_logging_and_dirs(experiment_name='experiment'):
    """设置日志和创建输出目录"""
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{experiment_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/stats", exist_ok=True)
    
    # 设置日志
    log_file = f"{output_dir}/logs/training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"实验输出目录: {output_dir}")
    
    return output_dir, logger

def save_training_stats(output_dir, stats_data, filename='training_stats.csv'):
    """保存训练统计数据到CSV"""
    stats_file = f"{output_dir}/stats/{filename}"
    file_exists = os.path.exists(stats_file)
    
    with open(stats_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats_data)
    
    return stats_file

def save_model_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False, prefix='model'):
    """保存模型checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存最新checkpoint
    checkpoint_file = f"{output_dir}/models/{prefix}_checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_file)
    
    # 保存最佳模型
    if is_best:
        best_model_file = f"{output_dir}/models/{prefix}_best.pt"
        torch.save(checkpoint, best_model_file)
        return best_model_file
    
    return checkpoint_file

def save_experiment_config(output_dir, config_dict, filename='config.json'):
    """保存实验配置"""
    config_file = f"{output_dir}/{filename}"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return config_file

def save_experiment_results(output_dir, results_dict, filename='results.json'):
    """保存实验结果汇总"""
    results_file = f"{output_dir}/{filename}"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    return results_file

def count_model_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 估算模型大小（MB）
    param_size = total_params * 4 / (1024 ** 2)  # 假设float32
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': param_size
    }

def get_model_summary(model, model_type):
    """获取模型详细信息"""
    param_info = count_model_parameters(model)
    
    summary = {
        'model_type': model_type,
        'parameters': param_info,
        'num_parameters_str': f"{param_info['total_params']:,}"
    }
    
    # 添加模型特定信息
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        # RNN或Transformer
        summary['architecture'] = 'Encoder-Decoder'
    
    if model_type == 'RNN':
        if hasattr(model, 'encoder'):
            enc = model.encoder
            summary['rnn_config'] = {
                'rnn_type': 'GRU',
                'layers': enc.rnn.num_layers,
                'hidden_dim': enc.rnn.hidden_size,
                'dropout': enc.rnn.dropout if hasattr(enc.rnn, 'dropout') else 0
            }
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'attention'):
            summary['attention_method'] = model.decoder.attention.method
    elif model_type == 'Transformer':
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            summary['transformer_config'] = {
                'n_layers': len(model.encoder.layers),
                'd_model': model.encoder.d_model,
                'n_heads': model.encoder.layers[0].self_attn.n_heads,
                'd_ff': model.encoder.layers[0].feed_forward.linear1.out_features,
                'pos_encoding': type(model.encoder.pos_encoding).__name__,
                'norm_type': type(model.encoder.layers[0].norm1).__name__
            }
    
    return summary

def log_experiment_setup(logger, config, model, model_type, dataset_info):
    """记录实验设置详情"""
    logger.info("="*80)
    logger.info("实验设置详情")
    logger.info("="*80)
    
    # 模型信息
    model_summary = get_model_summary(model, model_type)
    logger.info(f"\n【模型信息】")
    logger.info(f"  模型类型: {model_summary['model_type']}")
    logger.info(f"  架构: {model_summary.get('architecture', 'N/A')}")
    logger.info(f"  总参数量: {model_summary['num_parameters_str']} ({model_summary['parameters']['model_size_mb']:.2f} MB)")
    logger.info(f"  可训练参数: {model_summary['parameters']['trainable_params']:,}")
    logger.info(f"  不可训练参数: {model_summary['parameters']['non_trainable_params']:,}")
    
    if 'rnn_config' in model_summary:
        cfg = model_summary['rnn_config']
        logger.info(f"  RNN配置: 类型={cfg['rnn_type']}, 层数={cfg['layers']}, "
                   f"隐藏维度={cfg['hidden_dim']}, Dropout={cfg['dropout']}")
        logger.info(f"  Attention方法: {model_summary['attention_method']}")
    elif 'transformer_config' in model_summary:
        cfg = model_summary['transformer_config']
        logger.info(f"  Transformer配置: 层数={cfg['n_layers']}, d_model={cfg['d_model']}, "
                   f"注意力头={cfg['n_heads']}, d_ff={cfg['d_ff']}")
        logger.info(f"  位置编码: {cfg['pos_encoding']}")
        logger.info(f"  归一化: {cfg['norm_type']}")
    
    # 训练策略
    logger.info(f"\n【训练策略】")
    logger.info(f"  训练轮数: {config.get('n_epochs', 'N/A')}")
    logger.info(f"  批次大小: {config.get('batch_size', 'N/A')}")
    logger.info(f"  学习率: {config.get('lr', 'N/A')}")
    if 'dropout' in config:
        logger.info(f"  Dropout: {config['dropout']}")
    if 'tf_ratio' in config:
        logger.info(f"  Teacher Forcing比率: {config['tf_ratio']}")
    
    # 优化器
    logger.info(f"\n【优化器配置】")
    logger.info(f"  优化器类型: Adam")
    logger.info(f"  梯度裁剪: max_norm=1.0")
    
    # 解码策略
    logger.info(f"\n【解码策略】")
    logger.info(f"  解码方法: Beam Search")
    logger.info(f"  Beam Width: 5")
    logger.info(f"  最大解码长度: {config.get('max_len', 50)}")
    
    # 数据集信息
    logger.info(f"\n【数据集信息】")
    logger.info(f"  训练样本数: {dataset_info.get('train_samples', 'N/A')}")
    logger.info(f"  验证样本数: {dataset_info.get('valid_samples', 'N/A')}")
    logger.info(f"  源语言词表大小: {dataset_info.get('src_vocab_size', 'N/A')}")
    logger.info(f"  目标语言词表大小: {dataset_info.get('tgt_vocab_size', 'N/A')}")
    
    # 硬件信息
    logger.info(f"\n【硬件信息】")
    logger.info(f"  设备: {config.get('device', 'N/A')}")
    logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU数量: {torch.cuda.device_count()}")
        logger.info(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    logger.info("="*80)

def log_training_summary(logger, model_type, training_history, total_time, config):
    """记录训练汇总信息"""
    logger.info("\n" + "="*80)
    logger.info("训练汇总")
    logger.info("="*80)
    
    # 计算统计数据
    epoch_times = [epoch['time'] for epoch in training_history]
    avg_epoch_time = np.mean(epoch_times)
    min_epoch_time = np.min(epoch_times)
    max_epoch_time = np.max(epoch_times)
    
    # 收敛分析
    losses = [epoch['loss'] for epoch in training_history]
    bleus = [epoch['bleu4'] for epoch in training_history]
    
    initial_loss = losses[0] if losses else 0
    final_loss = losses[-1] if losses else 0
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    
    initial_bleu = bleus[0] if bleus else 0
    final_bleu = bleus[-1] if bleus else 0
    bleu_improvement = final_bleu - initial_bleu
    
    # 最佳结果
    best_epoch = max(training_history, key=lambda x: x['bleu4']) if training_history else None
    
    # 输出汇总
    logger.info(f"\n【训练时间分析】")
    logger.info(f"  总训练时间: {total_time / 60:.2f} 分钟 ({total_time / 3600:.2f} 小时)")
    logger.info(f"  平均每epoch时间: {avg_epoch_time:.2f} 秒")
    logger.info(f"  最快epoch时间: {min_epoch_time:.2f} 秒")
    logger.info(f"  最慢epoch时间: {max_epoch_time:.2f} 秒")
    logger.info(f"  总训练epochs: {len(training_history)}")
    
    logger.info(f"\n【收敛分析】")
    logger.info(f"  初始Loss: {initial_loss:.4f}")
    logger.info(f"  最终Loss: {final_loss:.4f}")
    logger.info(f"  Loss降低: {loss_reduction:.2f}%")
    logger.info(f"  初始BLEU-4: {initial_bleu:.4f}")
    logger.info(f"  最终BLEU-4: {final_bleu:.4f}")
    logger.info(f"  BLEU提升: {bleu_improvement:+.4f}")
    
    if best_epoch:
        logger.info(f"\n【最佳结果】")
        logger.info(f"  最佳Epoch: {best_epoch['epoch']}")
        logger.info(f"  最佳Loss: {best_epoch['loss']:.4f}")
        logger.info(f"  最佳BLEU-4: {best_epoch['bleu4']:.4f}")
        # 仅在存在precision指标时输出（某些模型类型可能不包含这些指标）
        if 'precision_1' in best_epoch:
            logger.info(f"  最佳P-1: {best_epoch['precision_1']:.4f}")
        if 'precision_2' in best_epoch:
            logger.info(f"  最佳P-2: {best_epoch['precision_2']:.4f}")
        if 'precision_3' in best_epoch:
            logger.info(f"  最佳P-3: {best_epoch['precision_3']:.4f}")
        if 'precision_4' in best_epoch:
            logger.info(f"  最佳P-4: {best_epoch['precision_4']:.4f}")
        if 'ppl' in best_epoch:
            logger.info(f"  最佳PPL: {best_epoch['ppl']:.2f}")
    
    logger.info("="*80)

# ==========================================
# 1. 核心评估函数：计算 BLEU-4 与 Precision_n
# ==========================================

def calculate_metrics(model, dataset, tgt_itos, beam_width=5, num_samples=200, max_len=50, decode_method=None, logger=None):
    """
    计算项目要求的指标：BLEU-1, BLEU-2, BLEU-3, BLEU-4, Precision-1, Precision-2, Precision-3, Precision-4

    Args:
        decode_method: 'greedy'（快速）或'beam'（慢但质量高）
                    如果为None，则根据模型类型自动选择
    """
    if logger:
        if decode_method:
            logger.debug(f"开始评估，使用{num_samples}个样本，decode_method={decode_method}，max_len={max_len}")
        else:
            logger.debug(f"开始评估，使用{num_samples}个样本，beam_width={beam_width}，max_len={max_len}")
    """
    计算项目要求的指标：BLEU-1, BLEU-2, BLEU-3, BLEU-4, Precision-1, Precision-2, Precision-3, Precision-4
    """
    model.eval()
    smoothie = SmoothingFunction().method1

    # 指标累加器 - 添加各阶BLEU
    total_bleu1 = 0
    total_bleu2 = 0
    total_bleu3 = 0
    total_bleu4 = 0
    total_precisions = {1: 0, 2: 0, 3: 0, 4: 0}

    # 随机采样进行评估
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for i in indices:
            src_ids, tgt_ids = dataset[i]
            src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

            # 1. 解码获得预测序列
            # 智能选择解码方法
            if decode_method == 'greedy' and hasattr(model, 'translate_greedy'):
                # Transformer使用贪婪解码
                pred_ids = model.translate_greedy(src_tensor, max_len=max_len)
            elif decode_method == 'greedy':
                # RNN使用beam_width=1（等价于贪婪）
                pred_ids = model.translate_beam(src_tensor, beam_width=1, max_len=max_len)
            elif decode_method == 'beam' or decode_method is None:
                # 使用Beam Search
                pred_ids = model.translate_beam(src_tensor, beam_width=beam_width, max_len=max_len)
            else:
                raise ValueError(f"Unknown decode_method: {decode_method}")
            
            # 2. 转换为 tokens 列表并移除 <SOS>, <EOS>, <PAD>
            pred_tokens = [tgt_itos.get(idx, '<UNK>') for idx in pred_ids if idx not in [0, 1, 2]]
            ref_tokens = [tgt_itos.get(idx, '<UNK>') for idx in tgt_ids if idx not in [0, 1, 2]]
            
            # 跳过空序列
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                continue
            
            # 3. 计算 BLEU-1, BLEU-2, BLEU-3, BLEU-4（使用不同权重）
            try:
                # BLEU-1: 只使用1-gram，权重=(1.0, 0, 0, 0)
                bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1.0, 0, 0, 0),
                                      smoothing_function=smoothie)
                total_bleu1 += bleu1
                
                # BLEU-2: 使用1-2 gram等权重，weights=(0.5, 0.5, 0, 0)
                bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0),
                                      smoothing_function=smoothie)
                total_bleu2 += bleu2
                
                # BLEU-3: 使用1-3 gram等权重，weights=(0.33, 0.33, 0.33, 0)
                bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                                      smoothing_function=smoothie)
                total_bleu3 += bleu3
                
                # BLEU-4: 使用1-4 gram等权重，weights=(0.25, 0.25, 0.25, 0.25)
                bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), 
                                      smoothing_function=smoothie)
                total_bleu4 += bleu4
            except:
                pass
            
            # 4. 计算各阶 Precision_n
            for n in range(1, 5):
                try:
                    p_n = modified_precision([ref_tokens], pred_tokens, n)
                    p_val = float(p_n.numerator) / p_n.denominator if p_n.denominator > 0 else 0
                    total_precisions[n] += p_val
                except:
                    pass
                
    # 取平均值
    valid_samples = len([i for i in indices if len([tgt_itos.get(idx, '<UNK>') for idx in dataset[i][1] if idx not in [0, 1, 2]]) > 0])
    if valid_samples == 0:
        return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    
    avg_bleu_scores = {
        1: total_bleu1 / valid_samples,
        2: total_bleu2 / valid_samples,
        3: total_bleu3 / valid_samples,
        4: total_bleu4 / valid_samples
    }
    avg_precisions = {n: val / valid_samples for n, val in total_precisions.items()}
    
    return avg_bleu_scores, avg_precisions

# ==========================================
# 2. RNN模型训练
# ==========================================

def train_rnn_model(attn_method='concat', tf_ratio=0.5, n_epochs=10, batch_size=64, lr=0.001, 
                     beam_width=5, max_len=50, eval_samples=200, output_dir=None, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_dir is None:
        output_dir, _ = setup_logging_and_dirs('rnn')
    
    # 保存配置
    config = {
        'model_type': 'RNN',
        'attn_method': attn_method,
        'tf_ratio': tf_ratio,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'max_len': max_len,
        'device': str(DEVICE),
        'beam_width': beam_width,
        'eval_samples': eval_samples
    }
    save_experiment_config(output_dir, config)
    
    # 模型初始化（使用更大的参数以提升性能）
    emb_dim = 512  # 增大embedding维度
    hid_dim = 1024  # 增大隐藏层维度
    n_layers = 3  # 增加层数
    enc = Encoder(len(src_vocab), emb_dim, hid_dim, n_layers=n_layers)
    dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, n_layers=n_layers, attn_method=attn_method)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # 记录实验设置详情
    dataset_info = {
        'train_samples': len(train_src),
        'valid_samples': len(valid_src),
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab)
    }
    log_experiment_setup(logger, config, model, 'RNN', dataset_info)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_loader = DataLoader(TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab), 
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    best_bleu = 0.0
    training_history = []
    start_time_total = time.time()
    
    logger.info(f"\n开始训练，使用Teacher Forcing策略，ratio={tf_ratio}")
    logger.info(f"解码策略: Beam Search (width={beam_width}, max_len={max_len})")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, trg, tf_ratio)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg_flat = trg.transpose(0, 1)[1:].reshape(-1)
            
            loss = criterion(output, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # 计算 Perplexity
        avg_loss = epoch_loss / len(train_loader)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        # 每个 Epoch 结束后观测 Metric（使用命令行参数）
        bleu_scores, precisions = calculate_metrics(model, valid_ds, tgt_itos,
                                                     beam_width=args.beam_width,
                                                     max_len=args.max_len,
                                                     num_samples=args.eval_samples,
                                                     decode_method=args.decode_method)

        epoch_time = time.time() - start_time
        is_best = bleu_scores[4] > best_bleu
        if is_best:
            best_bleu = bleu_scores[4]

        # 记录统计数据
        stats = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'ppl': ppl,
            'bleu4': bleu_scores[4],
            'bleu1': bleu_scores[1],
            'bleu2': bleu_scores[2],
            'bleu3': bleu_scores[3],
            'precision_1': precisions[1],
            'precision_2': precisions[2],
            'precision_3': precisions[3],
            'precision_4': precisions[4],
            'time': epoch_time,
            'is_best': is_best,
            'learning_rate': lr,
            'teacher_forcing_ratio': tf_ratio
        }
        training_history.append(stats)
        save_training_stats(output_dir, stats, 'training_stats.csv')
        
        # 保存模型checkpoint
        metrics = {'bleu4': bleu_scores[4], 'bleu1': bleu_scores[1], 'bleu2': bleu_scores[2], 'bleu3': bleu_scores[3], 'loss': avg_loss, 'ppl': ppl, **precisions}
        save_model_checkpoint(model, optimizer, epoch + 1, metrics, output_dir, is_best=is_best, prefix='rnn')
        
        logger.info(f"Epoch: {epoch+1:02}/{n_epochs:02} | Loss: {avg_loss:.3f} | PPL: {ppl:.2f}")
        logger.info(f"Metrics: BLEU-1: {bleu_scores[1]:.4f} | BLEU-2: {bleu_scores[2]:.4f} | BLEU-3: {bleu_scores[3]:.4f} | BLEU-4: {bleu_scores[4]:.4f}")
        logger.info(f"         P-1: {precisions[1]:.4f} | P-2: {precisions[2]:.4f} | P-3: {precisions[3]:.4f} | P-4: {precisions[4]:.4f}")
        logger.info(f"Time: {epoch_time:.2f}s | Best: {'✓' if is_best else '✗'}")
    
    total_time = time.time() - start_time_total
    
    # 保存最终结果
    results = {
        'best_bleu4': best_bleu,
        'training_history': training_history,
        'final_metrics': training_history[-1] if training_history else {},
        'total_training_time': total_time,
        'avg_epoch_time': np.mean([e['time'] for e in training_history]) if training_history else 0
    }
    save_experiment_results(output_dir, results)
    
    # 记录训练汇总
    log_training_summary(logger, 'RNN', training_history, total_time, config)
    logger.info(f"\n训练完成！最佳BLEU-4: {best_bleu:.4f}")

# ==========================================
# 3. Transformer模型训练（从零开始）
# ==========================================

def train_transformer_from_scratch(d_model=512, n_heads=8, n_layers=6, d_ff=2048,
                                   pos_type='absolute', norm_type='layernorm',
                                   n_epochs=10, batch_size=64, lr=0.0001, dropout=0.1,
                                   beam_width=5, max_len=50, eval_samples=200,
                                   decode_method='greedy', output_dir=None, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_dir is None:
        output_dir, _ = setup_logging_and_dirs('transformer')
    
    # 保存配置
    config = {
        'model_type': 'Transformer',
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'pos_type': pos_type,
        'norm_type': norm_type,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'dropout': dropout,
        'max_len': max_len,
        'device': str(DEVICE),
        'beam_width': beam_width,
        'eval_samples': eval_samples
    }
    save_experiment_config(output_dir, config)
    
    # 模型初始化
    # 注意：序列会添加<SOS>和<EOS>标记，所以max_len需要考虑这些标记
    model = TransformerNMT(
        len(src_vocab), len(tgt_vocab), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_len=MAX_LEN + 10, dropout=dropout,  # 增加10个位置编码的空间
        pos_type=pos_type, norm_type=norm_type, device=DEVICE
    ).to(DEVICE)
    
    # 记录实验设置详情
    dataset_info = {
        'train_samples': len(train_src),
        'valid_samples': len(valid_src),
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab)
    }
    log_experiment_setup(logger, config, model, 'Transformer', dataset_info)
    
    # 使用Adam优化器，带warmup
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_loader = DataLoader(TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab), 
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    best_bleu = 0.0
    training_history = []
    start_time_total = time.time()
    
    # 根据decode_method显示正确的解码策略
    if decode_method == 'greedy':
        logger.info(f"\n开始训练，解码策略: Greedy Decoding (max_len={max_len})")
    elif decode_method == 'beam':
        logger.info(f"\n开始训练，解码策略: Beam Search (width={beam_width}, max_len={max_len})")
    else:
        logger.info(f"\n开始训练，解码策略: {decode_method} (max_len={max_len})")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # Transformer使用teacher forcing（输入是trg[:-1]，目标是trg[1:]）
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, trg_input)  # [batch, tgt_len-1, vocab_size]
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_flat = trg_output.contiguous().view(-1)
            
            loss = criterion(output, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # 计算 Perplexity
        avg_loss = epoch_loss / len(train_loader)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        # 每个 Epoch 结束后观测 Metric
        bleu_scores, precisions = calculate_metrics(model, valid_ds, tgt_itos,
                                                   beam_width=beam_width,
                                                   max_len=max_len,
                                                   num_samples=eval_samples,
                                                   decode_method=decode_method,
                                                   logger=logger)

        epoch_time = time.time() - start_time
        is_best = bleu_scores[4] > best_bleu
        if is_best:
            best_bleu = bleu_scores[4]

        # 记录统计数据
        stats = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'ppl': ppl,
            'bleu4': bleu_scores[4],
            'bleu1': bleu_scores[1],
            'bleu2': bleu_scores[2],
            'bleu3': bleu_scores[3],
            'precision_1': precisions[1],
            'precision_2': precisions[2],
            'precision_3': precisions[3],
            'precision_4': precisions[4],
            'time': epoch_time,
            'is_best': is_best,
            'learning_rate': lr,
            'position_encoding': pos_type,
            'normalization': norm_type
        }
        training_history.append(stats)
        save_training_stats(output_dir, stats, 'training_stats.csv')
        
        # 保存模型checkpoint
        metrics = {'bleu4': bleu_scores[4], 'bleu1': bleu_scores[1], 'bleu2': bleu_scores[2], 'bleu3': bleu_scores[3], 'loss': avg_loss, 'ppl': ppl, **precisions}
        save_model_checkpoint(model, optimizer, epoch + 1, metrics, output_dir, is_best=is_best, prefix='transformer')
        
        logger.info(f"Epoch: {epoch+1:02}/{n_epochs:02} | Loss: {avg_loss:.3f} | PPL: {ppl:.2f}")
        logger.info(f"Metrics: BLEU-1: {bleu_scores[1]:.4f} | BLEU-2: {bleu_scores[2]:.4f} | BLEU-3: {bleu_scores[3]:.4f} | BLEU-4: {bleu_scores[4]:.4f}")
        logger.info(f"         P-1: {precisions[1]:.4f} | P-2: {precisions[2]:.4f} | P-3: {precisions[3]:.4f} | P-4: {precisions[4]:.4f}")
        logger.info(f"Time: {epoch_time:.2f}s | Best: {'✓' if is_best else '✗'}")
    
    total_time = time.time() - start_time_total
    
    # 保存最终结果
    results = {
        'best_bleu4': best_bleu,
        'training_history': training_history,
        'final_metrics': training_history[-1] if training_history else {},
        'total_training_time': total_time,
        'avg_epoch_time': np.mean([e['time'] for e in training_history]) if training_history else 0
    }
    save_experiment_results(output_dir, results)
    
    # 记录训练汇总
    log_training_summary(logger, 'Transformer', training_history, total_time, config)
    logger.info(f"\n训练完成！最佳BLEU-4: {best_bleu:.4f}")
    
    return model, best_bleu

# ==========================================
# 4. 架构消融实验
# ==========================================

def run_architectural_ablation(output_dir=None, logger=None):
    """比较不同的位置编码和归一化方法"""
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_dir is None:
        output_dir, logger = setup_logging_and_dirs('ablation')
    
    logger.info("\n" + "="*60)
    logger.info("架构消融实验：位置编码和归一化方法对比")
    logger.info("="*60)
    
    results = {}
    
    # 实验配置
    configs = [
        {'pos_type': 'absolute', 'norm_type': 'layernorm', 'name': 'Absolute+LayerNorm'},
        {'pos_type': 'absolute', 'norm_type': 'rmsnorm', 'name': 'Absolute+RMSNorm'},
        {'pos_type': 'relative', 'norm_type': 'layernorm', 'name': 'Relative+LayerNorm'},
        {'pos_type': 'relative', 'norm_type': 'rmsnorm', 'name': 'Relative+RMSNorm'},
    ]
    
    for config in configs:
        logger.info(f"\n实验配置: {config['name']}")
        exp_dir = f"{output_dir}/{config['name'].replace('+', '_')}"
        os.makedirs(exp_dir, exist_ok=True)
        model, best_bleu = train_transformer_from_scratch(
            d_model=256, n_heads=4, n_layers=3, d_ff=1024,
            pos_type=config['pos_type'], norm_type=config['norm_type'],
            n_epochs=5, batch_size=32, lr=0.0001,
            output_dir=exp_dir, logger=logger
        )
        results[config['name']] = best_bleu
        logger.info(f"{config['name']} 最佳BLEU-4: {best_bleu:.4f}")
    
    # 保存汇总结果
    summary = {
        'experiment_type': 'architectural_ablation',
        'results': results,
        'best_config': max(results.items(), key=lambda x: x[1])[0] if results else None
    }
    save_experiment_results(output_dir, summary, 'ablation_summary.json')
    
    logger.info("\n" + "="*60)
    logger.info("架构消融实验结果汇总:")
    logger.info("="*60)
    for name, bleu in results.items():
        logger.info(f"{name}: {bleu:.4f}")
    
    return results

# ==========================================
# 5. 超参数敏感性实验
# ==========================================

def run_hyperparameter_sensitivity(output_dir=None, logger=None):
    """测试不同超参数的影响"""
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_dir is None:
        output_dir, logger = setup_logging_and_dirs('hyperparameter_sensitivity')
    
    logger.info("\n" + "="*60)
    logger.info("超参数敏感性实验")
    logger.info("="*60)
    
    results = {}
    
    # 1. Batch Size 敏感性
    logger.info("\n1. Batch Size 敏感性实验")
    batch_sizes = [16, 32, 64, 128]
    for bs in batch_sizes:
        logger.info(f"\nBatch Size = {bs}")
        exp_dir = f"{output_dir}/batch_size_{bs}"
        os.makedirs(exp_dir, exist_ok=True)
        model, best_bleu = train_transformer_from_scratch(
            d_model=256, n_heads=4, n_layers=3, d_ff=1024,
            pos_type='absolute', norm_type='layernorm',
            n_epochs=5, batch_size=bs, lr=0.0001,
            output_dir=exp_dir, logger=logger
        )
        results[f'batch_size_{bs}'] = best_bleu
        logger.info(f"Batch Size {bs} 最佳BLEU-4: {best_bleu:.4f}")
    
    # 2. Learning Rate 敏感性
    logger.info("\n2. Learning Rate 敏感性实验")
    learning_rates = [0.00001, 0.0001, 0.001, 0.01]
    for lr in learning_rates:
        logger.info(f"\nLearning Rate = {lr}")
        exp_dir = f"{output_dir}/lr_{lr}"
        os.makedirs(exp_dir, exist_ok=True)
        model, best_bleu = train_transformer_from_scratch(
            d_model=256, n_heads=4, n_layers=3, d_ff=1024,
            pos_type='absolute', norm_type='layernorm',
            n_epochs=5, batch_size=32, lr=lr,
            output_dir=exp_dir, logger=logger
        )
        results[f'lr_{lr}'] = best_bleu
        logger.info(f"Learning Rate {lr} 最佳BLEU-4: {best_bleu:.4f}")
    
    # 3. Model Scale 敏感性
    logger.info("\n3. Model Scale 敏感性实验")
    model_scales = [
        {'d_model': 128, 'n_heads': 2, 'n_layers': 2, 'd_ff': 512, 'name': 'Small'},
        {'d_model': 256, 'n_heads': 4, 'n_layers': 3, 'd_ff': 1024, 'name': 'Medium'},
        {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'd_ff': 2048, 'name': 'Large'},
    ]
    for scale in model_scales:
        logger.info(f"\nModel Scale = {scale['name']}")
        exp_dir = f"{output_dir}/scale_{scale['name']}"
        os.makedirs(exp_dir, exist_ok=True)
        model, best_bleu = train_transformer_from_scratch(
            d_model=scale['d_model'], n_heads=scale['n_heads'],
            n_layers=scale['n_layers'], d_ff=scale['d_ff'],
            pos_type='absolute', norm_type='layernorm',
            n_epochs=5, batch_size=32, lr=0.0001,
            output_dir=exp_dir, logger=logger
        )
        results[f'scale_{scale["name"]}'] = best_bleu
        logger.info(f"Model Scale {scale['name']} 最佳BLEU-4: {best_bleu:.4f}")
    
    # 保存汇总结果
    summary = {
        'experiment_type': 'hyperparameter_sensitivity',
        'results': results,
        'best_batch_size': max([(bs, results[f'batch_size_{bs}']) for bs in batch_sizes], key=lambda x: x[1])[0] if results else None,
        'best_lr': max([(lr, results[f'lr_{lr}']) for lr in learning_rates], key=lambda x: x[1])[0] if results else None,
        'best_scale': max([(scale['name'], results[f'scale_{scale["name"]}']) for scale in model_scales], key=lambda x: x[1])[0] if results else None
    }
    save_experiment_results(output_dir, summary, 'hyperparameter_summary.json')
    
    logger.info("\n" + "="*60)
    logger.info("超参数敏感性实验结果汇总:")
    logger.info("="*60)
    for name, bleu in results.items():
        logger.info(f"{name}: {bleu:.4f}")
    
    return results

# ==========================================
# 6. 预训练模型微调（T5）
# ==========================================

def train_from_pretrained_t5(n_epochs=5, batch_size=32, lr=5e-5, src_itos=None, tgt_itos=None, 
                              train_src=None, train_tgt=None, valid_ds=None, device=None, output_dir=None, logger=None):
    """使用预训练的T5模型进行微调"""
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_dir is None:
        output_dir, logger = setup_logging_and_dirs('t5_finetune')
    
    logger.info("\n" + "="*60)
    logger.info("预训练模型微调：T5")
    logger.info("="*60)
    
    # 确定设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        import transformers
        from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
        logger.info(f"transformers版本: {transformers.__version__}")
        
        # 保存配置
        config = {
            'model_type': 'T5',
            'pretrained_model': 't5-small',
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'device': str(device)
        }
        save_experiment_config(output_dir, config)
        
        # 加载预训练T5模型和tokenizer
        logger.info("加载预训练T5模型...")
        model_name = 't5-small'  # 可以使用 't5-base' 或 't5-large' 如果有足够资源
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # 准备数据（需要将数据转换为T5格式）
        logger.info("准备训练数据...")
        if train_src is None or train_tgt is None:
            raise ValueError("train_src和train_tgt参数不能为None，请确保传入训练数据")
        
        train_texts = []
        train_labels = []
        
        for src_tokens, tgt_tokens in zip(train_src[:1000], train_tgt[:1000]):  # 使用部分数据
            src_text = ' '.join(src_tokens)
            tgt_text = ' '.join(tgt_tokens)
            train_texts.append(f"translate Chinese to English: {src_text}")
            train_labels.append(tgt_text)
        
        # Tokenize数据
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, max_length=128, return_tensors='pt')
        
        # 创建DataLoader
        class T5Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': self.labels['input_ids'][idx]
                }
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        train_dataset = T5Dataset(train_encodings, train_labels_encodings)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和学习率调度器
        # 使用torch.optim.AdamW代替transformers.AdamW（新版本transformers已移除AdamW）
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        num_training_steps = len(train_loader) * n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        # 记录实验设置详情
        dataset_info = {
            'train_samples': len(train_dataset),
            'valid_samples': len(valid_ds),
            'src_vocab_size': 'N/A (T5 tokenizer)',
            'tgt_vocab_size': 'N/A (T5 tokenizer)'
        }
        log_experiment_setup(logger, config, model, 'T5 (Pretrained)', dataset_info)
        
        # 训练循环
        best_bleu = 0.0
        training_history = []
        start_time_total = time.time()
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch: {epoch+1:02}/{n_epochs:02} | Loss: {avg_loss:.3f} | PPL: {ppl:.2f} | Time: {epoch_time:.2f}s")
            
            # 评估：计算BLEU-4和Precision指标
            bleu4 = 0.0
            bleu_scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
            precisions = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
            if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:  # 每2个epoch或最后一个epoch评估
                logger.info("开始验证...")
                model.eval()
                eval_samples = min(200, len(valid_ds))  # 使用更多样本以获得更准确的评估
                smoothie = SmoothingFunction().method1
                
                total_bleu_scores = {1: 0, 2: 0, 3: 0, 4: 0}
                total_precisions = {1: 0, 2: 0, 3: 0, 4: 0}
                valid_count = 0
                
                with torch.no_grad():
                    for i in range(eval_samples):
                        src_ids, tgt_ids = valid_ds[i]
                        if src_itos is not None:
                            src_text = ' '.join([src_itos.get(idx, '<UNK>') for idx in src_ids if idx not in [0, 1, 2]])
                        else:
                            src_text = ' '.join([str(idx) for idx in src_ids if idx not in [0, 1, 2]])
                        input_text = f"translate Chinese to English: {src_text}"
                        
                        input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=128, truncation=True).to(device)
                        outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
                        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        if tgt_itos is not None:
                            ref_text = ' '.join([tgt_itos.get(idx, '<UNK>') for idx in tgt_ids if idx not in [0, 1, 2]])
                        else:
                            ref_text = ' '.join([str(idx) for idx in tgt_ids if idx not in [0, 1, 2]])
                        
                        try:
                            pred_tokens = pred_text.split()
                            ref_tokens = ref_text.split()
                            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                                # 计算BLEU-1到BLEU-4（使用不同权重）
                                # BLEU-1: 只使用1-gram，weights=(1.0, 0, 0, 0)
                                bleu1_score = sentence_bleu([ref_tokens], pred_tokens, weights=(1.0, 0, 0, 0),
                                                           smoothing_function=smoothie)
                                total_bleu_scores[1] += bleu1_score
                                
                                # BLEU-2: 使用1-2 gram等权重，weights=(0.5, 0.5, 0, 0)
                                bleu2_score = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0),
                                                           smoothing_function=smoothie)
                                total_bleu_scores[2] += bleu2_score
                                
                                # BLEU-3: 使用1-3 gram等权重，weights=(0.33, 0.33, 0.33, 0)
                                bleu3_score = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                                                           smoothing_function=smoothie)
                                total_bleu_scores[3] += bleu3_score
                                
                                # BLEU-4: 使用1-4 gram等权重，weights=(0.25, 0.25, 0.25, 0.25)
                                bleu4_score = sentence_bleu([ref_tokens], pred_tokens, 
                                                           weights=(0.25, 0.25, 0.25, 0.25),
                                                           smoothing_function=smoothie)
                                total_bleu_scores[4] += bleu4_score
                                
                                # 计算各阶Precision
                                for n in range(1, 5):
                                    p_n = modified_precision([ref_tokens], pred_tokens, n)
                                    p_val = float(p_n.numerator) / p_n.denominator if p_n.denominator > 0 else 0
                                    total_precisions[n] += p_val
                                
                                valid_count += 1
                        except Exception as e:
                            # 跳过计算失败的样本
                            continue
                
                if valid_count > 0:
                    # 计算平均BLEU分数
                    bleu_scores = {n: total_bleu_scores[n] / valid_count for n in range(1, 5)}
                    bleu4 = bleu_scores[4]
                    precisions = {n: total_precisions[n] / valid_count for n in range(1, 5)}
                    logger.info(f"Validation BLEU-1: {bleu_scores[1]:.4f} | BLEU-2: {bleu_scores[2]:.4f} | BLEU-3: {bleu_scores[3]:.4f} | BLEU-4: {bleu4:.4f}")
                    logger.info(f"Validation P-1: {precisions[1]:.4f} | P-2: {precisions[2]:.4f} | P-3: {precisions[3]:.4f} | P-4: {precisions[4]:.4f}")
                    
                    if bleu4 > best_bleu:
                        best_bleu = bleu4
                        logger.info(f"发现新的最佳模型！BLEU-4: {best_bleu:.4f}")
                        # 保存最佳模型
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'bleu4': best_bleu,
                            'bleu1': bleu_scores[1],
                            'bleu2': bleu_scores[2],
                            'bleu3': bleu_scores[3],
                            'precision_1': precisions[1],
                            'precision_2': precisions[2],
                            'precision_3': precisions[3],
                            'precision_4': precisions[4],
                            'loss': avg_loss,
                            'timestamp': datetime.now().isoformat()
                        }
                        torch.save(checkpoint, f"{output_dir}/models/t5_best.pt")
            
            # 记录统计数据
            stats = {
                'epoch': epoch + 1,
                'loss': avg_loss,
                'ppl': ppl,
                'bleu1': bleu_scores[1],
                'bleu2': bleu_scores[2],
                'bleu3': bleu_scores[3],
                'bleu4': bleu_scores[4],
                'precision_1': precisions[1],
                'precision_2': precisions[2],
                'precision_3': precisions[3],
                'precision_4': precisions[4],
                'time': epoch_time,
                'is_best': bleu_scores[4] == best_bleu,
                'learning_rate': lr
            }
            training_history.append(stats)
            save_training_stats(output_dir, stats, 'training_stats.csv')
        
        total_time = time.time() - start_time_total
        
        # 保存最终结果
        results = {
            'best_bleu4': best_bleu,
            'training_history': training_history,
            'final_metrics': training_history[-1] if training_history else {},
            'total_training_time': total_time,
            'avg_epoch_time': np.mean([e['time'] for e in training_history]) if training_history else 0
        }
        save_experiment_results(output_dir, results)
        
        # 记录训练汇总
        if training_history:
            log_training_summary(logger, 'T5 (Pretrained)', training_history, total_time, config)
        
        logger.info(f"\nT5微调最佳BLEU-4: {best_bleu:.4f}")
        return model, best_bleu
        
    except ImportError as e:
        if logger:
            logger.error(f"导入transformers库失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error("请检查transformers库是否正确安装:")
            logger.error("  pip install transformers")
            logger.error("或者检查Python环境和依赖项")
        else:
            print(f"错误: 导入transformers库失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            print("请检查transformers库是否正确安装: pip install transformers")
        return None, 0.0
    except Exception as e:
        import traceback
        if logger:
            logger.error(f"T5微调过程中出错: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error("详细错误信息:")
            logger.error(traceback.format_exc())
        else:
            print(f"T5微调过程中出错: {e}")
            print(f"错误类型: {type(e).__name__}")
            traceback.print_exc()
        return None, 0.0

# ==========================================
# 7. 主函数
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NMT模型训练和评估')
    parser.add_argument('--model', type=str, default='transformer', choices=['rnn', 'transformer', 'ablation', 'hyperparam', 't5', 'all'],
                        help='模型类型或实验类型')
    parser.add_argument('--data', type=str, default='train_10k.jsonl', help='训练数据文件')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='层数')
    parser.add_argument('--pos_type', type=str, default='absolute', choices=['absolute', 'relative'], help='位置编码类型')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='归一化类型')
    
    # 训练策略参数
    parser.add_argument('--tf_ratio', type=float, default=0.5, help='Teacher Forcing比例（RNN专用，0-1之间）')
    parser.add_argument('--attn_method', type=str, default='concat', choices=['dot', 'general', 'concat'],
                        help='Attention方法（RNN专用：dot点积/general乘法/concat加法）')
    
    # 解码策略参数
    parser.add_argument('--decode_method', type=str, default='greedy', choices=['greedy', 'beam'],
                        help='解码方法：greedy（贪婪，快速）、beam（Beam Search，较慢但质量更高）')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam Search宽度（仅当decode_method=beam时有效）')
    parser.add_argument('--max_len', type=int, default=50, help='最大解码长度')
    parser.add_argument('--eval_samples', type=int, default=200, help='评估时使用的样本数')
    
    args = parser.parse_args()
    
    # 设置日志和输出目录
    output_dir, logger = setup_logging_and_dirs(args.model)
    logger.info("="*60)
    logger.info("NMT模型训练和评估")
    logger.info("="*60)
    logger.info(f"实验类型: {args.model}")
    logger.info(f"参数: {vars(args)}")
    
    # 数据准备
    logger.info("加载数据...")
    train_src, train_tgt = process_file(f'dataset/{args.data}', max_len=args.max_len)
    valid_src, valid_tgt = process_file('dataset/valid.jsonl', max_len=args.max_len)
    src_vocab, src_itos = build_vocab(train_src)
    tgt_vocab, tgt_itos = build_vocab(train_tgt)
    
    valid_ds = TranslationDataset(valid_src, valid_tgt, src_vocab, tgt_vocab)
    
    N_EPOCHS = args.epochs
    MAX_LEN = args.max_len
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"训练样本数: {len(train_src)}, 验证样本数: {len(valid_src)}")
    logger.info(f"源语言词表大小: {len(src_vocab)}, 目标语言词表大小: {len(tgt_vocab)}")
    
    # 根据参数选择实验
    if args.model == 'rnn':
        train_rnn_model(attn_method=args.attn_method, tf_ratio=args.tf_ratio, 
                       n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       beam_width=args.beam_width, max_len=args.max_len, eval_samples=args.eval_samples,
                       output_dir=output_dir, logger=logger)
    elif args.model == 'transformer':
        train_transformer_from_scratch(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            pos_type=args.pos_type, norm_type=args.norm_type,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            beam_width=args.beam_width, max_len=args.max_len, eval_samples=args.eval_samples,
            decode_method=args.decode_method, output_dir=output_dir, logger=logger
        )
    elif args.model == 'ablation':
        run_architectural_ablation(output_dir=output_dir, logger=logger)
    elif args.model == 'hyperparam':
        run_hyperparameter_sensitivity(output_dir=output_dir, logger=logger)
    elif args.model == 't5':
        train_from_pretrained_t5(n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                 src_itos=src_itos, tgt_itos=tgt_itos,
                                 train_src=train_src, train_tgt=train_tgt, valid_ds=valid_ds,
                                 device=DEVICE, output_dir=output_dir, logger=logger)
    elif args.model == 'all':
        logger.info("运行所有实验...")
        # 1. Transformer从零开始训练
        logger.info("\n实验1: Transformer从零开始训练")
        exp1_dir = f"{output_dir}/transformer_scratch"
        os.makedirs(exp1_dir, exist_ok=True)
        train_transformer_from_scratch(
            d_model=256, n_heads=4, n_layers=3, d_ff=1024,
            pos_type='absolute', norm_type='layernorm',
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            beam_width=args.beam_width, max_len=args.max_len, eval_samples=args.eval_samples,
            decode_method=args.decode_method, output_dir=exp1_dir, logger=logger
        )
        # 2. 架构消融
        exp2_dir = f"{output_dir}/ablation"
        os.makedirs(exp2_dir, exist_ok=True)
        run_architectural_ablation(output_dir=exp2_dir, logger=logger)
        # 3. 超参数敏感性
        exp3_dir = f"{output_dir}/hyperparameter"
        os.makedirs(exp3_dir, exist_ok=True)
        run_hyperparameter_sensitivity(output_dir=exp3_dir, logger=logger)
        # 4. T5微调
        exp4_dir = f"{output_dir}/t5_finetune"
        os.makedirs(exp4_dir, exist_ok=True)
        train_from_pretrained_t5(n_epochs=5, batch_size=32, lr=5e-5, 
                                 src_itos=src_itos, tgt_itos=tgt_itos,
                                 train_src=train_src, train_tgt=train_tgt, valid_ds=valid_ds,
                                 device=DEVICE, output_dir=exp4_dir, logger=logger)
    
    logger.info(f"\n所有实验完成！结果保存在: {output_dir}")
