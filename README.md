# 基于BERT与GRU的网络谣言识别研究

## 项目简介

本项目实现了一个基于 **BERT + 双向GRU + 注意力机制** 的深度学习模型，用于中文网络谣言的自动检测与识别。

### 模型架构

```
BERT (bert-base-chinese) → 双向GRU → 注意力机制 → LayerNorm → 全连接分类
```

- **BERT**: 使用中文预训练模型提取深层语义特征
- **双向GRU**: 捕捉文本序列的双向依赖关系
- **注意力机制**: 聚焦关键信息，提升模型判别能力
- **LayerNorm + Dropout**: 稳定训练，防止过拟合

### 项目结构

```
├── model.py              # 模型定义 (BERT+GRU+Attention)
├── train.py              # 训练脚本
├── predict.py            # 命令行预测脚本
├── app.py                # Flask Web应用
├── templates/
│   └── index.html        # Web界面
├── data/
│   ├── train25.csv       # 训练数据 (CSV格式)
│   ├── train.txt         # 训练数据 (TXT格式)
│   ├── test.csv          # 测试数据
│   ├── test.txt          # 测试数据
│   └── all_data.txt      # 完整数据集
├── requirements.txt      # Python依赖
└── README.md             # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认CSV数据训练
python train.py

# 使用TXT格式数据训练
python train.py --data_path data/train.txt --txt_format

# 使用完整数据集训练 (更多epoch)
python train.py --data_path data/all_data.txt --txt_format --epoch 30 --batch_size 32
```

训练参数说明:
- `--data_path`: 训练数据路径
- `--epoch`: 训练轮数 (默认20)
- `--batch_size`: 批次大小 (默认16)
- `--lr`: 学习率 (默认2e-4)
- `--dropout`: Dropout比例 (默认0.5)
- `--val_split`: 验证集比例 (默认0.2)

### 3. 模型预测

```bash
# 单条文本预测
python predict.py --text "今天在芙蓉溪发现一具男尸，死者被割喉挖肾"

# 批量文件预测
python predict.py --input data/test.csv --output results.xlsx
```

### 4. 启动Web应用

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000`，即可使用Web界面进行谣言检测。

## 数据格式

### CSV格式
| 语言/文本/评论 | 标签 |
|:---|:---|
| 这是一条谣言... | 0 |
| 这是正常信息... | 1 |

- 文本列名支持: `语言`、`文本`、`评论`
- 标签: `0` = 谣言, `1` = 非谣言

### TXT格式
```
0	这是一条谣言文本...
1	这是正常信息文本...
```

每行: `标签<TAB>文本内容`

## 技术特点

1. **深度语义理解**: 利用BERT中文预训练模型提取深层语义特征
2. **动态时序捕捉**: 双向GRU网络捕捉文本的长距离依赖关系
3. **注意力聚焦**: 注意力机制自动学习关键信息权重
4. **可复现性**: 固定随机种子，保证实验结果可复现
5. **多格式支持**: 支持CSV、Excel、TXT等多种数据格式

## 依赖环境

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Flask 2.3+
- Pandas 1.5+

首次运行时会自动从HuggingFace下载 `bert-base-chinese` 预训练模型。
