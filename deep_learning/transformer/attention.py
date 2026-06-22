"""
Attention（注意力机制）
=======================

本文件从零实现 Transformer 的核心：**Scaled Dot-Product Attention（缩放点积注意力）**
与 **Multi-Head Attention（多头注意力）**。

------------------------------------------------------------
一、Scaled Dot-Product Attention
------------------------------------------------------------
给定 Query（查询）、Key（键）、Value（值）三组向量：

        Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

直觉：用 Q 与每个 K 做点积衡量“相关性”，softmax 归一化成权重后，
对 V 加权求和。除以 √d_k 是为了防止维度变大时点积过大、softmax 进入
梯度极小的饱和区（即“缩放”的由来）。

形状（shape）约定（B=batch, H=heads, L=序列长度, d_k=每个头的维度）：
        Q: (B, H, L_q, d_k)
        K: (B, H, L_k, d_k)
        V: (B, H, L_k, d_v)
        输出: (B, H, L_q, d_v)，注意力权重: (B, H, L_q, L_k)

掩码（mask）：
    - padding mask：屏蔽 <pad> 位置，避免对填充符做注意力；
    - look-ahead / causal mask：解码器自注意力中屏蔽“未来”位置，
      保证自回归生成时第 t 步看不到 t 之后的信息。
    被屏蔽的位置在 softmax 之前置为一个极大负数（softmax 后≈0）。

------------------------------------------------------------
二、Multi-Head Attention
------------------------------------------------------------
单一注意力只能学到一种“关注模式”。多头注意力把 d_model 维拆成 h 个头，
每个头在低维子空间里独立做注意力，最后拼接并线性映射回 d_model：

        head_i   = Attention(Q·W_iQ, K·W_iK, V·W_iV)
        MultiHead = Concat(head_1, ..., head_h) · W_O

让模型在不同子空间、不同位置上同时关注多种信息。

依赖：仅使用 PyTorch 张量与基础算子，注意力计算全部手写。
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """缩放点积注意力。

    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    参数
    ----
    query : Tensor, shape (..., L_q, d_k)
    key   : Tensor, shape (..., L_k, d_k)
    value : Tensor, shape (..., L_k, d_v)
    mask  : Tensor 或 None，可广播到 (..., L_q, L_k)；值为 0/False 的位置会被屏蔽
    dropout : nn.Dropout 或 None，作用在注意力权重上（论文中注意力权重也会 dropout）

    返回
    ----
    output  : Tensor, shape (..., L_q, d_v)
    attn    : Tensor, shape (..., L_q, L_k)，softmax 后的注意力权重（便于可视化/调试）
    """
    # d_k 是每个头的维度，用它来做缩放
    d_k = query.size(-1)

    # 1) 打分：Q 与每个 K 做点积。key 的最后两维转置，使 (.., L_q, d_k)·(.., d_k, L_k)
    #    得到 (.., L_q, L_k)，即“每个 query 对每个 key 的相关性分数”。
    #    除以 √d_k 做缩放，防止点积随维度增大而过大、把 softmax 推向饱和区。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2) 掩码：把不该被关注的位置（mask==0）填成极大负数。
    #    softmax 后这些位置的权重≈0。这里用 -1e9 这样的“大负数”而非 -inf，
    #    是为了避免“某一行被全部屏蔽”时 softmax 出现 NaN（全 -inf 会得到 0/0）。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3) 归一化成概率分布：对最后一维（所有 key）做 softmax。
    attn = torch.softmax(scores, dim=-1)

    # 4)（可选）对注意力权重做 dropout，作为正则化。
    if dropout is not None:
        attn = dropout(attn)

    # 5) 用注意力权重对 V 加权求和，得到每个 query 位置的输出表示。
    output = torch.matmul(attn, value)
    return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力。

    把 d_model 拆成 num_heads 个头，每个头维度 d_k = d_model // num_heads，
    各自做缩放点积注意力后拼接，再经 W_O 线性映射回 d_model。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q/K/V 各一个线性投影，输出仍是 d_model（= num_heads * d_k，相当于把所有头拼在一起算）。
        # 输出投影 W_O 把拼接后的多头结果再融合回 d_model 维。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        # 保存最近一次的注意力权重，方便外部可视化/调试（不参与训练）。
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        参数
        ----
        query : (B, L_q, d_model)
        key   : (B, L_k, d_model)
        value : (B, L_k, d_model)
        mask  : 可广播到 (B, 1, L_q, L_k) 的张量；约定掩码已含“头”这一维（dim=1），
                因此这里无需再为头扩维（见 transformer.py 中的 make_*_mask）。

        返回
        ----
        (B, L_q, d_model)

        说明：自注意力时 query=key=value 同源；交叉注意力时 query 来自解码器、
              key/value 来自编码器输出。
        """
        B = query.size(0)

        # 1) 线性投影后“分头”：(B, L, d_model) --view--> (B, L, H, d_k) --transpose--> (B, H, L, d_k)
        #    transpose(1,2) 把“头”维提到序列维前面，使每个头都能并行做注意力。
        #    这里用 -1 让 view 自动推断序列长度（q、k、v 的长度可以不同）。
        q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) 在每个头上做缩放点积注意力。out: (B, H, L_q, d_k)，attn: (B, H, L_q, L_k)
        out, self.attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3) “合头”：把 (B, H, L_q, d_k) 转回 (B, L_q, H, d_k) 再 reshape 成 (B, L_q, d_model)。
        #    transpose 后内存不连续，需 contiguous() 才能 view。
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 4) 输出投影，融合各头信息。
        return self.w_o(out)
