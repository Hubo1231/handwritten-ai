"""
Transformer 最简使用示例
========================

用一个**玩具复制任务 (copy task)** 验证 Transformer 实现是否正确：
让模型学会把源序列原样“复制”到输出。这是调试 seq2seq 模型最经典的
sanity check——数据零成本、规律极简，几百步就能收敛，便于快速确认
前向 / 掩码 / 训练循环写得对不对。

约定的特殊 token：
    0 = <pad>（填充，本例定长所以用不到，但保留以演示 padding mask）
    1 = <bos>（句子起始，解码器的第一个输入）
    2..vocab_size-1 = 普通 token

teacher forcing（教师强制）：训练时把“正确答案右移一位”喂给解码器，
即解码器在第 t 步看到的是真实的第 t 个 token，去预测第 t+1 个。
这样可以并行训练整条序列（配合因果掩码保证不偷看未来）。

运行：
    python example.py
"""

import torch
import torch.nn as nn

from transformer import Transformer, make_pad_mask, make_causal_mask

PAD_IDX = 0
BOS_IDX = 1


def make_batch(batch_size, seq_len, vocab_size):
    """生成一批“复制任务”样本。

    src     : (B, L)      随机源序列（token 取 2..vocab_size-1，避开 pad/bos）
    tgt_in  : (B, L)      解码器输入 = [<bos>, src_0, ..., src_{L-2}]
    tgt_out : (B, L)      期望输出   = [src_0, src_1, ..., src_{L-1}]  即原样复制 src
    """
    src = torch.randint(2, vocab_size, (batch_size, seq_len))
    bos = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long)
    tgt_in = torch.cat([bos, src[:, :-1]], dim=1)  # 右移一位，开头补 <bos>
    tgt_out = src                                  # 目标就是把 src 复制出来
    return src, tgt_in, tgt_out


def build_masks(src, tgt_in):
    """构造源/目标掩码。"""
    src_mask = make_pad_mask(src, PAD_IDX)                       # (B,1,1,L_src)
    tgt_pad = make_pad_mask(tgt_in, PAD_IDX)                     # (B,1,1,L_tgt)
    tgt_causal = make_causal_mask(tgt_in.size(1)).to(tgt_in.device)  # (1,1,L_tgt,L_tgt)
    tgt_mask = tgt_pad & tgt_causal                             # padding ∧ 因果，广播成 (B,1,L_tgt,L_tgt)
    return src_mask, tgt_mask


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len):
    """贪心解码：每步取概率最大的 token，自回归地逐个生成。

    这正是“推理时”与训练的关键区别——没有标准答案可喂，只能用模型
    自己上一步的输出作为下一步的输入。
    """
    model.eval()
    memory = model.encode(src, src_mask)                 # 源序列只需编码一次
    ys = torch.full((src.size(0), 1), BOS_IDX, dtype=torch.long, device=src.device)  # 从 <bos> 开始
    for _ in range(max_len):
        tgt_mask = make_causal_mask(ys.size(1)).to(src.device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])             # 只取最后一个位置的预测
        next_tok = logits.argmax(dim=-1, keepdim=True)   # 贪心：取最大概率
        ys = torch.cat([ys, next_tok], dim=1)            # 追加到已生成序列
    return ys[:, 1:]                                     # 去掉开头的 <bos>


def train():
    # ---- 超参数（玩具规模，CPU 几秒即可跑）----
    vocab_size = 12
    seq_len = 8
    batch_size = 32
    d_model, num_heads, num_layers, d_ff = 64, 4, 2, 128
    steps = 1000  # 玩具复制任务约 800~1000 步可达 100% 准确率

    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )
    # ignore_index=PAD_IDX：损失不计 <pad> 位置（本例定长用不到，习惯性写上）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out = make_batch(batch_size, seq_len, vocab_size)
        src_mask, tgt_mask = build_masks(src, tgt_in)

        logits = model(src, tgt_in, src_mask, tgt_mask)   # (B, L, vocab)
        # CrossEntropyLoss 期望 (N, C) 与 (N,)，所以把 batch 和序列维拍平。
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # ---- 训练完用贪心解码验证“复制”效果 ----
    src, _, _ = make_batch(1, seq_len, vocab_size)
    src_mask = make_pad_mask(src, PAD_IDX)
    pred = greedy_decode(model, src, src_mask, max_len=seq_len)
    print("\n=== 复制任务验证 ===")
    print("源序列  src :", src[0].tolist())
    print("模型输出 pred:", pred[0].tolist())
    print("完全复制成功 ✅" if torch.equal(pred, src) else "仍有错误，可增大 steps 再试")

    return model


if __name__ == "__main__":
    train()
