"""
BERT + GRU + Attention 网络谣言检测 — Web应用
==============================================
基于Flask的Web界面，支持:
  - 单条文本输入检测
  - CSV/Excel/TXT文件批量上传检测
  - 结果下载为Excel

启动方式:
  python app.py
  然后访问 http://127.0.0.1:5000
"""

import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
from transformers import BertTokenizer

from model import RumorDetectionModel

app = Flask(__name__)
CORS(app)

# ========== 配置 ==========
MODEL_PATH = os.environ.get('MODEL_PATH', 'rumor_detection_model.pth')
BERT_MODEL_NAME = os.environ.get('BERT_MODEL_NAME', 'bert-base-chinese')
MAX_LEN = 128

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_PATH}")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

model = RumorDetectionModel(dropout=0.5)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
)
model.eval()

if torch.cuda.is_available():
    model.cuda()
    print("使用CUDA进行推理")
else:
    print("使用CPU进行推理")


def prepare_data(texts, max_len=MAX_LEN):
    """文本预处理：tokenize"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return inputs["input_ids"], inputs["attention_mask"]


def detect_rumor(texts):
    """
    对文本列表进行谣言检测

    参数:
        texts: list[str], 待检测的文本列表

    返回:
        predictions: tensor, 预测结果 (0=谣言, 1=非谣言)
    """
    input_ids, attention_mask = prepare_data(texts)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """谣言检测API"""
    try:
        texts = []

        # 1. 获取直接输入的文本
        text_input = request.form.get('text')
        if text_input and text_input.strip():
            texts.append(text_input.strip())

        # 2. 处理文件上传
        file = request.files.get('file')
        if file and file.filename:
            filename = file.filename
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

            if ext in ['xls', 'xlsx', 'csv']:
                # CSV/Excel文件
                try:
                    if ext == 'csv':
                        data = pd.read_csv(file)
                    else:
                        data = pd.read_excel(file)
                except Exception as e:
                    return jsonify({
                        'message': '文件读取失败，请检查文件格式是否正确。',
                        'error': str(e)
                    }), 400

                # 查找文本列
                text_col = None
                for col in ['文本', '评论', '语言']:
                    if col in data.columns:
                        text_col = col
                        break

                if text_col is None:
                    return jsonify({
                        'message': f'文件中缺少文本列。支持的列名: 文本、评论、语言。当前列: {list(data.columns)}'
                    }), 400

                texts.extend(data[text_col].dropna().astype(str).tolist())

            elif ext == 'txt':
                # TXT文件 — 每行一条文本
                content = file.read().decode('utf-8')
                texts.extend([line.strip() for line in content.split('\n') if line.strip()])

            else:
                return jsonify({
                    'message': f'不支持的文件格式: .{ext}。支持的格式: CSV, Excel (.xls/.xlsx), TXT'
                }), 400

        # 3. 检查是否有文本
        if not texts:
            return jsonify({'message': '请上传文件或输入文本。'}), 400

        # 4. 检测
        predictions = detect_rumor(texts)
        results = [
            {
                '句子': text[:200] + ('...' if len(text) > 200 else ''),
                '标签': '谣言' if pred.item() == 0 else '非谣言'
            }
            for text, pred in zip(texts, predictions)
        ]

        # 5. 保存结果
        result_df = pd.DataFrame(results)
        result_file = 'result.xlsx'
        result_df.to_excel(result_file, index=False)

        rumor_count = sum(1 for r in results if r['标签'] == '谣言')
        non_rumor_count = len(results) - rumor_count

        return jsonify({
            'message': f'检测成功! 共 {len(results)} 条: 谣言 {rumor_count} 条, 非谣言 {non_rumor_count} 条',
            'downloadUrl': url_for('download_file', filename='result.xlsx'),
            'summary': {
                'total': len(results),
                'rumor': rumor_count,
                'non_rumor': non_rumor_count
            }
        })

    except Exception as e:
        return jsonify({
            'message': '检测失败，请重试。',
            'error': str(e)
        }), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载结果文件"""
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    print("=" * 50)
    print("网络谣言检测系统启动中...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
