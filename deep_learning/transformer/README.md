# Transformer

从零手写实现 **Transformer**（Vaswani et al., 2017，《Attention Is All You Need》），
深度学习现代架构的基石。本目录按模块拆分，逐个手写注意力、位置编码与编码器/解码器。

> 状态：✅ 已实现并验证（`python example.py` 玩具复制任务约 1000 步可达 100% 准确率）。
> 代码含逐行中文注释讲解，建议对照下方原理边读边跑。

## 📂 文件说明

| 文件 | 说明 |
| --- | --- |
| `attention.py` | 缩放点积注意力 + 多头注意力 |
| `positional_encoding.py` | 正弦位置编码 |
| `transformer.py` | 前馈网络、编码器层 / 解码器层、Encoder/Decoder 堆叠、完整 Transformer、掩码工具 |
| `example.py` | 最简使用示例：玩具复制任务（copy task）训练循环 |
| `notebooks/` | 配套手写讲解 notebook（手撕 Attention / 手搓 Transformer 等） |
| `reference/` | 论文原文、逐段精读 PDF，以及 `ai-by-hand-excel` 可视化资料 |
| `data/` | 数据集（如 multi30k）；已在 `.gitignore` 中忽略 |

## 🧮 算法原理

### 1. Scaled Dot-Product Attention（缩放点积注意力）

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

用 Q 与每个 K 的点积衡量相关性，softmax 归一化为权重后对 V 加权求和；
除以 `√d_k` 防止维度增大时点积过大导致 softmax 梯度饱和。

### 2. Multi-Head Attention（多头注意力）

```
head_i    = Attention(Q·W_iQ, K·W_iK, V·W_iV)
MultiHead = Concat(head_1, ..., head_h) · W_O
```

把 `d_model` 拆成 `h` 个头，在不同子空间并行关注多种模式。

### 3. Positional Encoding（位置编码）

自注意力对位置无感，需注入位置信息。原论文采用固定正弦编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. Encoder & Decoder（编码器与解码器）

- **EncoderLayer** = 多头自注意力 + 前馈网络，每个子层带残差 + LayerNorm。
- **DecoderLayer** = 掩码自注意力 + 交叉注意力（Q 来自解码器、K/V 来自编码器输出）+ 前馈网络。
- 前馈网络：`FFN(x) = relu(x·W_1 + b_1)·W_2 + b_2`。

掩码：`src_mask` 屏蔽 `<pad>`；`tgt_mask` = padding mask ∧ 因果（look-ahead）mask，
保证自回归生成时看不到未来 token。

## 🗺️ 阅读顺序（建议）

1. `attention.py` → `scaled_dot_product_attention` → `MultiHeadAttention`
2. `positional_encoding.py` → `PositionalEncoding`
3. `transformer.py` → `PositionwiseFeedForward` → `EncoderLayer` / `DecoderLayer`
   → `Encoder` / `Decoder` → `Transformer` → 掩码工具（`make_pad_mask` / `make_causal_mask`）
4. `example.py` 跑通玩具复制任务，并用 `greedy_decode` 观察自回归生成。

## 🚀 快速开始

```bash
# 安装依赖
pip install torch

# 运行示例：训练玩具复制任务，并贪心解码验证“复制”效果
python example.py
```

## 📚 参考资料

- 论文原文：`reference/attention is all you need.pdf`
- 逐段精读：`reference/【论文精读 & 面试题】Transformer论文逐段精读.pdf`
- 可视化：`reference/ai-by-hand-excel/`（Excel 手算 Attention / Transformer）
