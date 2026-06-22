"""
Positional Encoding（位置编码）
==============================

自注意力本身对位置“无感”——打乱输入顺序，输出只是相应地被打乱，
模型无法区分“我爱你”和“你爱我”。因此需要把**位置信息**注入到词向量中。

------------------------------------------------------------
正弦位置编码 (Sinusoidal Positional Encoding)
------------------------------------------------------------
原论文 "Attention Is All You Need" 使用固定的正余弦函数：

        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i /
        .3d_model) )

其中 pos 是位置（0,1,2,...），i 是维度索引。

为什么这样设计：
    - 不同维度对应不同波长，从 2π 到 10000·2π，形成由密到疏的频率谱；
    - 任意固定偏移 k，PE(pos+k) 都能表示成 PE(pos) 的线性函数，
      便于模型学习“相对位置”关系；
    - 无需训练参数，且能外推到比训练时更长的序列。

使用：把 PE 直接**加到**词嵌入上（二者维度同为 d_model），再送入编码器/解码器。

形状（shape）：
        输入  x : (B, L, d_model)
        输出    : (B, L, d_model)   # x + PE[:L]

依赖：仅使用 PyTorch 张量算子手写。
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码。

    预先计算好 (max_len, d_model) 的位置编码表并缓存为 buffer
    （非训练参数），前向时取前 L 行加到输入上。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预先把 (max_len, d_model) 的位置编码表算好，避免每次前向都重算。
        # 约定 d_model 为偶数（sin/cos 各占一半维度）。
        pe = torch.zeros(max_len, d_model)

        # position: 每个位置的下标 0,1,2,...,max_len-1，形状 (max_len, 1) 便于广播。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: 公式中的 1 / 10000^(2i/d_model)，对应不同维度的“频率”。
        # 用 exp(log(...)) 的等价写法在数值上更稳定：
        #   10000^(-2i/d_model) = exp( 2i · (-log(10000)/d_model) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # 形状 (d_model/2,)

        # 偶数维用 sin，奇数维用 cos（position 与 div_term 广播成 (max_len, d_model/2)）。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维 -> (1, max_len, d_model)，便于和 (B, L, d_model) 的输入相加。
        pe = pe.unsqueeze(0)

        # register_buffer: pe 会随模型 .to(device)/保存加载一起走，但不是可训练参数。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        参数
        ----
        x : (B, L, d_model)  词嵌入

        返回
        ----
        (B, L, d_model)  加上位置编码后的结果
        """
        # 取位置编码表的前 L 行，加到词嵌入上（位置信息以“加法”方式注入）。
        # self.pe 形状 (1, max_len, d_model)，切片后 (1, L, d_model) 广播到 batch。
        # 用 self.pe[:, :L] 而非整表，既支持变长序列，也省显存。
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
