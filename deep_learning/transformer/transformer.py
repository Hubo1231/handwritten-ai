"""
Transformer（编码器 - 解码器整体架构）
=====================================

本文件在 attention.py / positional_encoding.py 的基础上，组装出完整的
Transformer（Vaswani et al., 2017, "Attention Is All You Need"）。

------------------------------------------------------------
整体结构
------------------------------------------------------------
        源序列 ─► [输入嵌入 + 位置编码] ─► Encoder × N ─┐
                                                        ├─► 线性 + softmax ─► 预测
        目标序列 ─► [输入嵌入 + 位置编码] ─► Decoder × N ─┘

每个 EncoderLayer（编码器层）= 多头自注意力 + 前馈网络，
每个子层都包裹 **残差连接 (residual) + 层归一化 (LayerNorm)**：

        x = LayerNorm( x + Sublayer(x) )       # Post-LN（原论文写法）

每个 DecoderLayer（解码器层）有三个子层：
    1. 带因果掩码的多头自注意力（masked self-attention）；
    2. 编码器-解码器交叉注意力（Q 来自解码器，K/V 来自编码器输出）；
    3. 前馈网络。

------------------------------------------------------------
Position-wise Feed-Forward（逐位置前馈网络）
------------------------------------------------------------
        FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2

对每个位置独立地做两层全连接（中间维度 d_ff 通常为 4·d_model），
为模型提供非线性变换能力。

------------------------------------------------------------
掩码（mask）约定
------------------------------------------------------------
本实现约定所有 mask 都已带上“头”这一维（dim=1），可直接广播到
注意力分数 (B, H, L_q, L_k)：
    - src_mask : 屏蔽源序列的 <pad>，形状 (B, 1, 1, L_src)
    - tgt_mask : padding mask 与因果（look-ahead）mask 的结合，形状 (B, 1, L_tgt, L_tgt)
mask 中值为 True/1 表示“保留”，False/0 表示“屏蔽”。

依赖：仅使用 PyTorch；注意力与前向逻辑全部手写。
"""

import copy
import math

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from positional_encoding import PositionalEncoding


def clones(module, n):
    """复制 n 份结构相同、参数独立的子模块，返回 nn.ModuleList。

    用 deepcopy 保证每一层都有自己独立的参数（而不是共享同一份）。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络：FFN(x) = relu(x·W_1 + b_1)·W_2 + b_2。

    “逐位置(position-wise)”指对序列里每个位置用同一组权重独立计算，
    位置之间不交互（位置间的信息交互由注意力负责）。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # 升维：d_model -> d_ff（通常 4×）
        self.w_2 = nn.Linear(d_ff, d_model)   # 降维：d_ff -> d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 先升维并经 ReLU 引入非线性，dropout 正则化，再降回 d_model。
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """单个编码器层：自注意力 + 前馈，各带残差与 LayerNorm（Post-LN）。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 两个子层各配一个 LayerNorm。LayerNorm 对每个 token 的 d_model 维做归一化，
        # 稳定训练；残差连接让梯度更易回传、缓解深层退化。
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """x: (B, L_src, d_model) -> (B, L_src, d_model)"""
        # 子层1：自注意力。q=k=v=x（自己看自己），src_mask 屏蔽 <pad>。
        # Post-LN 写法：LayerNorm(x + Dropout(Sublayer(x)))
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # 子层2：前馈网络。
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """单个解码器层：掩码自注意力 + 交叉注意力 + 前馈。"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)   # 带因果掩码的自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 交叉注意力：Q=解码器, K/V=编码器
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        x      : (B, L_tgt, d_model)  解码器输入（已嵌入+位置编码）
        memory : (B, L_src, d_model)  编码器输出
        src_mask : 屏蔽源端 <pad>（用于交叉注意力的 key）
        tgt_mask : 因果+padding 掩码（用于解码器自注意力）
        """
        # 子层1：掩码自注意力。tgt_mask 保证第 t 步看不到未来 token。
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # 子层2：交叉注意力。Query 来自解码器(x)，Key/Value 来自编码器输出(memory)，
        #         这是解码器“读取”源序列信息的通道。src_mask 屏蔽源端填充。
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        # 子层3：前馈网络。
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """N 个编码器层堆叠。"""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, src_mask=None):
        # 逐层传递，上一层的输出作为下一层的输入。
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """N 个解码器层堆叠。"""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 每一层都用到同一份编码器输出 memory。
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """完整的 Transformer：嵌入 + 位置编码 + 编码器 + 解码器 + 输出投影。"""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入：把 token id 映射成 d_model 维向量。源/目标各一套词表与嵌入。
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码（源/目标共用同一套正弦编码即可）。
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 编码器 / 解码器：各自堆叠 num_layers 层。
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), num_layers)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), num_layers)

        # 输出投影：把解码器输出映射到目标词表大小，得到每个位置的 logits。
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier 初始化：让各层输入/输出方差大致一致，训练更稳定（原论文做法）。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        # 嵌入后乘 √d_model（论文 3.4 节：放大嵌入量级，与位置编码量级匹配），再加位置编码。
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src : (B, L_src) 源 token id
        tgt : (B, L_tgt) 目标 token id（teacher forcing 时为右移后的目标）

        返回
        ----
        (B, L_tgt, tgt_vocab_size) 每个位置在目标词表上的 logits
        """
        memory = self.encode(src, src_mask)            # 编码源序列 -> (B, L_src, d_model)
        out = self.decode(tgt, memory, src_mask, tgt_mask)  # 解码 -> (B, L_tgt, d_model)
        return self.generator(out)                     # 投影到词表 logits


# ----------------------------------------------------------------------
# 掩码工具函数
#   约定：返回的 mask 已带 batch 维与“头”维（dim=1），可直接广播到
#         注意力分数 (B, H, L_q, L_k)。True=保留，False=屏蔽。
# ----------------------------------------------------------------------
def make_pad_mask(seq, pad_idx):
    """padding mask：标记非 <pad> 位置为 True。

    seq : (B, L) -> (B, 1, 1, L)
    两次 unsqueeze 补出“头”维(dim=1)与“query”维(dim=2)，
    使其能广播到 (B, H, L_q, L)——即对所有头、所有 query 屏蔽相同的 <pad> key。
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size):
    """因果（look-ahead）掩码：下三角为 True，屏蔽未来位置。

    返回 (1, 1, size, size) 的布尔下三角矩阵：第 i 行只有 0..i 列为 True，
    表示位置 i 只能注意到自己及之前的位置。
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)
