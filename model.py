"""
BERT + GRU + Attention 网络谣言检测模型
========================================
架构: BERT(中文) → GRU(双向) → Attention → LayerNorm → FC
用于中文网络谣言/非谣言的二分类任务。
"""

import torch
import torch.nn as nn
from transformers import BertModel


class Attention(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output):
        # gru_output: (batch_size, seq_len, hidden_dim*2)
        attn_weights = torch.softmax(self.attn(gru_output), dim=1)
        attn_output = torch.sum(attn_weights * gru_output, dim=1)
        return attn_output


class RumorDetectionModel(nn.Module):
    """
    BERT + 双向GRU + 注意力机制 + LayerNorm 的谣言检测模型

    参数:
        dropout: float, Dropout比例
        n_class: int, 分类类别数 (默认2: 谣言/非谣言)
    """
    def __init__(self, dropout=0.5, n_class=2, bert_model_name='bert-base-chinese'):
        super(RumorDetectionModel, self).__init__()

        # BERT层 — 使用中文预训练模型
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 双向GRU层 — 输入为BERT hidden_size + 2个额外特征
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size + 2,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 注意力机制
        self.attention = Attention(256 * 2)

        # 层归一化
        self.layer_norm = nn.LayerNorm(256 * 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接分类层
        self.fc = nn.Linear(256 * 2, n_class)

    def forward(self, input_ids, attention_mask, extra_features=None):
        """
        前向传播

        参数:
            input_ids: (batch_size, seq_len) token ids
            attention_mask: (batch_size, seq_len) attention mask
            extra_features: (batch_size, 2) 额外特征, 为None时自动补零

        返回:
            out: (batch_size, n_class) 分类logits
        """
        # BERT输出 (含hidden_states用于取最后两层CLS)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = bert_output.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 如果没有额外特征，创建全零向量
        if extra_features is None:
            batch_size = input_ids.size(0)
            extra_features = torch.zeros(batch_size, 2, device=input_ids.device)

        # 将额外特征扩展到序列维度
        extra_features = extra_features.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)

        # 拼接BERT输出与额外特征
        combined_input = torch.cat((last_hidden_state, extra_features), dim=-1)

        # 通过双向GRU
        gru_output, _ = self.gru(combined_input)

        # 注意力机制
        attn_output = self.attention(gru_output)

        # LayerNorm + Dropout
        norm_output = self.layer_norm(attn_output)
        out = self.dropout(norm_output)

        # 全连接分类
        out = self.fc(out)
        return out
