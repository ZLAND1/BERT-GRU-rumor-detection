"""
BERT + GRU + Attention 网络谣言检测 — 训练脚本
==============================================
支持CSV和TXT格式的训练数据，自动划分训练集/验证集。

CSV格式要求:
  - 列名需包含 "语言" 或 "文本" (文本列) 和 "标签" (标签列)
  - 标签: 0=谣言, 1=非谣言

TXT格式要求:
  - 每行: 标签\t文本内容
  - 标签: 0=谣言, 1=非谣言

用法:
  python train.py                          # 使用默认参数和data/train25.csv
  python train.py --data_path data/train.txt --txt_format
  python train.py --epoch 10 --lr 0.0001 --batch_size 32
"""

import argparse
import numpy as np
import random
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd

from model import RumorDetectionModel


def setup_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv_data(data_path):
    """
    加载CSV格式数据
    CSV需要包含文本列("语言"/"文本"/"评论")和标签列("标签")
    """
    data = pd.read_csv(data_path)

    # 查找文本列
    text_col = None
    for col in ['语言', '文本', '评论']:
        if col in data.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"CSV文件中未找到文本列。可用列: {list(data.columns)}")

    # 查找标签列
    if '标签' not in data.columns:
        raise ValueError(f"CSV文件中未找到'标签'列。可用列: {list(data.columns)}")

    texts = data[text_col].astype(str).tolist()
    labels = data['标签'].astype(int).tolist()

    return texts, labels


def load_txt_data(data_path):
    """
    加载TXT格式数据
    每行: 标签\t文本内容
    """
    texts = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                label = int(parts[0])
                text = parts[1]
                labels.append(label)
                texts.append(text)

    print(f"从TXT加载了 {len(texts)} 条数据")
    return texts, labels


def load_data(data_path, txt_format=False):
    """根据格式加载数据"""
    if txt_format or data_path.endswith('.txt'):
        return load_txt_data(data_path)
    elif data_path.endswith('.csv'):
        return load_csv_data(data_path)
    else:
        # 尝试CSV，失败则尝试TXT
        try:
            return load_csv_data(data_path)
        except Exception:
            return load_txt_data(data_path)


def prepare_data(texts, labels, tokenizer, max_len=128):
    """将文本转换为模型输入"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels


def train_epoch(model, x_ids, x_mask, y, optimizer, loss_function, args, use_gpu):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    indices = torch.randperm(len(x_ids))
    for i in range(0, len(x_ids), args.batch_size):
        batch_indices = indices[i:i + args.batch_size]
        batch_input_ids = x_ids[batch_indices]
        batch_attention_mask = x_mask[batch_indices]
        batch_labels = y[batch_indices]

        if use_gpu:
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_labels = batch_labels.cuda()

        optimizer.zero_grad()
        outputs = model(batch_input_ids, batch_attention_mask)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, x_ids, x_mask, y, loss_function, use_gpu):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        if use_gpu:
            x_ids = x_ids.cuda()
            x_mask = x_mask.cuda()
            y = y.cuda()

        outputs = model(x_ids, x_mask)
        loss = loss_function(outputs, y)
        pred_y = torch.max(outputs, 1)[1]
        accuracy = (pred_y == y).sum().item() / len(y)

    return loss.item(), accuracy


def main():
    parser = argparse.ArgumentParser(description='BERT+GRU+Attention 谣言检测模型训练')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/train25.csv',
                        help='训练数据路径 (CSV或TXT)')
    parser.add_argument('--txt_format', action='store_true',
                        help='以TXT格式加载数据 (标签\\t文本)')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')

    # 模型参数
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout比例')
    parser.add_argument('--n_class', type=int, default=2,
                        help='分类类别数')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--l2', type=float, default=0.01,
                        help='L2正则化系数 (weight decay)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=5,
                        help='随机种子')

    # 输出参数
    parser.add_argument('--save_path', type=str, default='rumor_detection_model.pth',
                        help='模型保存路径')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='BERT预训练模型名称')

    args = parser.parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        print("请确保数据文件路径正确。")
        return

    # 检测GPU
    use_gpu = torch.cuda.is_available()
    device_str = 'CUDA' if use_gpu else 'CPU'
    print(f"使用设备: {device_str}")
    print(f"数据路径: {args.data_path}")

    # 加载数据
    print("加载数据...")
    texts, labels = load_data(args.data_path, args.txt_format)
    print(f"总样本数: {len(texts)}")
    print(f"谣言样本数: {labels.count(0)}, 非谣言样本数: {labels.count(1)}")

    # 加载tokenizer
    print(f"加载BERT模型: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # 准备数据
    x_ids, x_mask, y = prepare_data(texts, labels, tokenizer, max_len=args.max_len)

    # 划分训练集/验证集
    n_val = int(len(x_ids) * args.val_split)
    n_train = len(x_ids) - n_val

    train_ids, val_ids = x_ids[:n_train], x_ids[n_train:]
    train_mask, val_mask = x_mask[:n_train], x_mask[n_train:]
    train_y, val_y = y[:n_train], y[n_train:]

    print(f"训练集: {n_train} 条, 验证集: {n_val} 条")

    # 初始化模型
    print("初始化模型...")
    model = RumorDetectionModel(dropout=args.dropout, n_class=args.n_class)
    if use_gpu:
        model = model.cuda()

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )
    loss_function = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0
    print(f"\n开始训练 ({args.epoch} epochs)...")
    print("-" * 60)

    for epoch in range(args.epoch):
        train_loss = train_epoch(
            model, train_ids, train_mask, train_y,
            optimizer, loss_function, args, use_gpu
        )
        val_loss, val_acc = evaluate(
            model, val_ids, val_mask, val_y,
            loss_function, use_gpu
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            saved = " *"
        else:
            saved = ""

        print(f"Epoch {epoch+1:3d}/{args.epoch} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}{saved}")

    print("-" * 60)
    print(f"训练完成! 最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存到: {args.save_path}")


if __name__ == "__main__":
    main()
