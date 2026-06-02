"""
BERT + GRU + Attention 网络谣言检测 — Web应用
==============================================
基于Flask的Web界面，支持:
  - 单条文本输入检测（含置信度）
  - CSV/Excel/TXT文件批量上传检测
  - 在线预览批量结果、统计图表数据
  - 结果下载为Excel

启动方式:
  python app.py
  然后访问 http://127.0.0.1:5000
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
from transformers import BertTokenizer

from model import RumorDetectionModel

app = Flask(__name__)
CORS(app)

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'rumor_detection_model.pth'))
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
    对文本列表进行谣言检测，返回预测类别和置信度

    返回:
        predictions: tensor, 预测结果 (0=谣言, 1=非谣言)
        confidences: tensor, 置信度 (softmax概率)
    """
    input_ids, attention_mask = prepare_data(texts)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu(), probs.cpu()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """谣言检测API — 支持单条文本和批量文件"""
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
                content = file.read().decode('utf-8')
                texts.extend([line.strip() for line in content.split('\n') if line.strip()])

            else:
                return jsonify({
                    'message': f'不支持的文件格式: .{ext}。支持的格式: CSV, Excel (.xls/.xlsx), TXT'
                }), 400

        if not texts:
            return jsonify({'message': '请上传文件或输入文本。'}), 400

        # 3. 检测
        predictions, confidences = detect_rumor(texts)
        results = []
        for text, pred, conf in zip(texts, predictions, confidences):
            label = '谣言' if pred.item() == 0 else '非谣言'
            confidence = round(conf[pred.item()].item() * 100, 2)
            results.append({
                '句子': text[:200] + ('...' if len(text) > 200 else ''),
                '全文': text,
                '标签': label,
                '预测类别': pred.item(),
                '置信度': confidence
            })

        # 4. 保存结果到绝对路径
        result_file = os.path.join(BASE_DIR, 'result.xlsx')
        result_df = pd.DataFrame([{
            '序号': i + 1,
            '文本': r['全文'],
            '预测标签': r['标签'],
            '置信度(%)': r['置信度']
        } for i, r in enumerate(results)])
        result_df.to_excel(result_file, index=False)

        rumor_count = sum(1 for r in results if r['预测类别'] == 0)
        non_rumor_count = len(results) - rumor_count

        # 批量结果只返回摘要+前20条预览
        preview = results[:20] if len(results) > 20 else results

        return jsonify({
            'message': f'检测完成：共 {len(results)} 条',
            'downloadUrl': url_for('download_file', filename='result.xlsx'),
            'summary': {
                'total': len(results),
                'rumor': rumor_count,
                'non_rumor': non_rumor_count,
                'rumorRate': round(rumor_count / len(results) * 100, 1) if results else 0
            },
            'results': preview,
            'hasMore': len(results) > 20,
            'totalCount': len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'message': '检测失败，请重试。',
            'error': str(e)
        }), 500


@app.route('/api/detect_single', methods=['POST'])
def detect_single():
    """单条文本快速检测API"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'message': '请输入文本'}), 400

        predictions, confidences = detect_rumor([text])
        pred = predictions[0].item()
        conf = confidences[0]
        label = '谣言' if pred == 0 else '非谣言'

        return jsonify({
            'text': text[:500],
            'label': label,
            'prediction': pred,
            'confidence': round(conf[pred].item() * 100, 2),
            'rumorProb': round(conf[0].item() * 100, 2),
            'nonRumorProb': round(conf[1].item() * 100, 2),
            'message': f'检测结果：{label}（置信度 {round(conf[pred].item() * 100, 1)}%）'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'message': '检测失败', 'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载结果文件（使用绝对路径）"""
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'message': '文件不存在，请重新检测'}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    print("=" * 55)
    print("  网络谣言检测系统  v2.0")
    print("  模型: BERT + BiGRU + Attention")
    print(f"  访问地址: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
