"""
BERT + GRU + Attention 网络谣言检测 — 命令行预测脚本
=====================================================
支持单条文本预测和批量文件预测。

用法:
  # 单条文本预测
  python predict.py --text "今天在芙蓉溪发现一具男尸"

  # 批量文件预测 (CSV/Excel/TXT)
  python predict.py --input data/test.csv --output results.xlsx

  # 从TXT文件预测
  python predict.py --input data/test.txt --output results.xlsx
"""

import argparse
import os
import sys

import torch
import pandas as pd
from transformers import BertTokenizer

from model import RumorDetectionModel


def load_model(model_path='rumor_detection_model.pth', bert_model='bert-base-chinese'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 python train.py 训练模型。")
        sys.exit(1)

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = RumorDetectionModel(dropout=0.5)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    )
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    return model, tokenizer


def prepare_data(texts, tokenizer, max_len=128):
    """文本预处理"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return inputs["input_ids"], inputs["attention_mask"]


def predict_texts(model, tokenizer, texts, max_len=128):
    """对文本列表进行预测"""
    input_ids, attention_mask = prepare_data(texts, tokenizer, max_len)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu().tolist()


def load_texts_from_file(input_path):
    """从文件加载文本列表"""
    ext = input_path.rsplit('.', 1)[-1].lower() if '.' in input_path else ''

    if ext == 'csv':
        data = pd.read_csv(input_path)
        for col in ['文本', '评论', '语言']:
            if col in data.columns:
                return data[col].dropna().astype(str).tolist()
        # 如果找不到指定列，使用第一列
        return data.iloc[:, 0].dropna().astype(str).tolist()

    elif ext in ['xls', 'xlsx']:
        data = pd.read_excel(input_path)
        for col in ['文本', '评论', '语言']:
            if col in data.columns:
                return data[col].dropna().astype(str).tolist()
        return data.iloc[:, 0].dropna().astype(str).tolist()

    elif ext == 'txt':
        with open(input_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"不支持的文件格式: .{ext}")


def main():
    parser = argparse.ArgumentParser(description='BERT+GRU+Attention 谣言检测预测')

    parser.add_argument('--text', type=str, default=None,
                        help='单条文本输入进行预测')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径 (CSV/Excel/TXT)')
    parser.add_argument('--output', type=str, default='predictions.xlsx',
                        help='输出结果文件路径 (默认: predictions.xlsx)')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rumor_detection_model.pth'),
                        help='训练好的模型路径')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese',
                        help='BERT模型名称')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')

    args = parser.parse_args()

    # 检查输入
    if args.text is None and args.input is None:
        parser.print_help()
        print("\n错误: 请指定 --text 或 --input 参数。")
        sys.exit(1)

    # 加载模型
    print("加载模型...")
    model, tokenizer = load_model(args.model_path, args.bert_model)
    print("模型加载完成。")

    # 单条文本预测
    if args.text:
        pred = predict_texts(model, tokenizer, [args.text], args.max_len)[0]
        label = '谣言' if pred == 0 else '非谣言'
        print(f"\n文本: {args.text}")
        print(f"预测结果: {label} (类别ID: {pred})")
        return

    # 批量文件预测
    if args.input:
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)

        print(f"加载数据: {args.input}")
        texts = load_texts_from_file(args.input)
        print(f"共 {len(texts)} 条文本")

        print("预测中...")
        predictions = predict_texts(model, tokenizer, texts, args.max_len)

        # 生成结果
        results = []
        for text, pred in zip(texts, predictions):
            results.append({
                '文本': text,
                '预测标签': '谣言' if pred == 0 else '非谣言',
                '预测类别': pred
            })

        df = pd.DataFrame(results)
        df.to_excel(args.output, index=False)

        rumor_count = sum(1 for r in results if r['预测类别'] == 0)
        non_rumor_count = len(results) - rumor_count

        print(f"\n预测完成!")
        print(f"  总数: {len(results)}")
        print(f"  谣言: {rumor_count}")
        print(f"  非谣言: {non_rumor_count}")
        print(f"  结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
