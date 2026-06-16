"""
REINFORCE (蒙特卡洛策略梯度 / Monte-Carlo Policy Gradient)
===========================================================

本文件从零实现经典的 REINFORCE 算法（Williams, 1992），属于
Policy Gradient（策略梯度）方法家族中最基础的一员。

------------------------------------------------------------
一、核心思想
------------------------------------------------------------
与基于价值的方法（如 Q-Learning / DQN）不同，策略梯度方法
**直接参数化策略本身** π_θ(a|s)，并通过梯度上升来优化期望回报：

        目标函数  J(θ) = E_{τ~π_θ} [ R(τ) ]

其中 τ = (s_0, a_0, s_1, a_1, ...) 是一条轨迹（trajectory），
R(τ) 是这条轨迹的总回报。我们希望调整参数 θ，使得能带来高回报
的轨迹出现的概率更大。

------------------------------------------------------------
二、策略梯度定理 (Policy Gradient Theorem)
------------------------------------------------------------
直接对 J(θ) 求梯度会遇到“回报本身不依赖 θ，但轨迹分布依赖 θ”
的问题。利用对数求导技巧 (log-derivative trick)：

        ∇_θ J(θ) = E_{τ~π_θ} [ ( Σ_t ∇_θ log π_θ(a_t|s_t) ) · R(τ) ]

REINFORCE 进一步用 **reward-to-go**（从 t 时刻往后的累计回报 G_t）
替换整条轨迹的总回报 R(τ)，因为 t 时刻的动作只能影响它之后的奖励，
这样可以降低梯度估计的方差：

        ∇_θ J(θ) = E [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]

        其中  G_t = Σ_{k=t}^{T} γ^{k-t} · r_k     （折扣回报）

------------------------------------------------------------
三、从梯度到“损失函数”
------------------------------------------------------------
PyTorch 的优化器默认做的是梯度 **下降**（最小化），而我们想做梯度
**上升**（最大化 J）。因此构造一个“伪损失”(surrogate loss)，对它
求最小化就等价于对 J 求最大化：

        L(θ) = - Σ_t log π_θ(a_t|s_t) · G_t

注意：这里的 L 并不是真正意义上的损失（它没有“真实标签”），
它只是一个其梯度恰好等于 -∇_θ J 的标量，方便交给 autograd。

------------------------------------------------------------
四、算法流程
------------------------------------------------------------
    1. 用当前策略 π_θ 与环境交互，采样一条完整轨迹（一个 episode）；
    2. 计算每个时间步的折扣回报 G_t；
    3.（可选）对 G_t 做基线 (baseline) 处理 / 标准化，以降低方差；
    4. 计算伪损失 L 并反向传播，更新参数 θ；
    5. 重复以上过程，直到策略收敛。

依赖：仅使用 PyTorch（搭建策略网络 + 自动求导）与 gymnasium（环境）。
策略梯度的核心计算（回报、损失、更新）全部手写，不依赖任何 RL 库。
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 策略网络 (Policy Network)
# ============================================================
class PolicyNetwork(nn.Module):
    """一个简单的多层感知机 (MLP)，用来参数化离散动作策略 π_θ(a|s)。

    输入：状态向量 s（维度 = state_dim）
    输出：每个动作的 logits（未归一化的对数概率，维度 = action_dim）

    我们输出 logits 而不是直接输出概率，原因是：
      - 数值更稳定（配合 Categorical / softmax 内部的 log-sum-exp）；
      - 便于后续直接构造分类分布 (Categorical) 来采样动作并求 log π。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回各动作的 logits。

        参数:
            state: 形状 (batch, state_dim) 或 (state_dim,) 的张量
        返回:
            logits: 形状 (batch, action_dim) 的张量
        """
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return logits


# ============================================================
# 2. REINFORCE 智能体 (Agent)
# ============================================================
class REINFORCEAgent:
    """封装策略网络、动作采样、回报计算与参数更新的智能体。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-2,
        gamma: float = 0.99,
        normalize_returns: bool = True,
        device: str = "cpu",
    ):
        """
        参数:
            state_dim:          状态空间维度
            action_dim:         离散动作的数量
            hidden_dim:         隐藏层宽度
            lr:                 学习率
            gamma:              折扣因子 γ ∈ [0, 1]，越接近 1 越看重长远回报
            normalize_returns:  是否对每个 episode 的回报做标准化（零均值、单位方差），
                                这是降低梯度方差最简单常用的技巧（充当一个 baseline）
            device:             "cpu" 或 "cuda"
        """
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.device = torch.device(device)

        # 策略网络与优化器
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # --------------------------------------------------------
    # 2.1 根据当前策略采样一个动作
    # --------------------------------------------------------
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """给定状态，从策略分布 π_θ(·|s) 中采样一个动作。

        返回:
            action:   采样得到的动作（int）
            log_prob: 该动作的对数概率 log π_θ(a|s)（带梯度的标量张量），
                      之后用于构造策略梯度的损失。
        """
        # 转成张量，形状 (state_dim,) -> (1, state_dim)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.policy(state_tensor)  # (1, action_dim)

        # 用 logits 构造分类分布。Categorical 内部会对 logits 做 softmax。
        dist = torch.distributions.Categorical(logits=logits)

        # 从分布中采样一个动作（这一步引入了探索）
        action = dist.sample()  # 形状 (1,)

        # log π_θ(a|s)，保留计算图以便反向传播
        log_prob = dist.log_prob(action)  # 形状 (1,)

        return int(action.item()), log_prob.squeeze(0)

    # --------------------------------------------------------
    # 2.2 计算折扣回报 (reward-to-go)
    # --------------------------------------------------------
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """根据一个 episode 的奖励序列，计算每个时间步的折扣回报 G_t。

            G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
                = r_t + γ·G_{t+1}

        采用从后往前的递推方式，时间复杂度 O(T)。

        参数:
            rewards: 长度为 T 的奖励列表 [r_0, r_1, ..., r_{T-1}]
        返回:
            returns: 形状 (T,) 的张量 [G_0, G_1, ..., G_{T-1}]
        """
        returns: List[float] = []
        G = 0.0
        # 从最后一个时间步往前递推
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)  # 插到最前面，保持时间顺序

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # 标准化：减均值、除以标准差。
        # 这相当于使用“整段回报的均值”作为 baseline，能显著降低梯度方差、
        # 让训练更稳定。加上 1e-8 防止除零。
        if self.normalize_returns and returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        return returns_tensor

    # --------------------------------------------------------
    # 2.3 用一个 episode 的数据更新策略
    # --------------------------------------------------------
    def update(self, log_probs: List[torch.Tensor], rewards: List[float]) -> float:
        """执行一次 REINFORCE 参数更新。

        参数:
            log_probs: 该 episode 中每一步的 log π_θ(a_t|s_t) 列表（带梯度）
            rewards:   该 episode 中每一步的即时奖励列表
        返回:
            loss_value: 本次更新的伪损失数值（仅用于日志记录）
        """
        # 计算每个时间步的折扣回报 G_t
        returns = self.compute_returns(rewards)  # (T,)

        # 把每一步的 log_prob 拼成一个张量 (T,)
        log_probs_tensor = torch.stack(log_probs)  # (T,)

        # 伪损失：L = - Σ_t log π_θ(a_t|s_t) · G_t
        # 对 L 做梯度下降 == 对期望回报 J 做梯度上升。
        loss = -(log_probs_tensor * returns).sum()

        # 标准的三步走：清零梯度 -> 反向传播 -> 更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


# ============================================================
# 3. 训练循环
# ============================================================
def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 600,
    gamma: float = 0.99,
    lr: float = 1e-2,
    hidden_dim: int = 128,
    seed: int = 0,
    log_interval: int = 20,
    solved_threshold: float | None = None,
) -> REINFORCEAgent:
    """在指定的 gymnasium 环境上训练 REINFORCE 智能体。

    参数:
        env_name:         gymnasium 环境名称（需为离散动作空间）
        num_episodes:     训练的 episode 总数
        gamma / lr / hidden_dim: 见 REINFORCEAgent
        seed:             随机种子，保证可复现
        log_interval:     每隔多少个 episode 打印一次日志
        solved_threshold: 当最近若干个 episode 的平均回报超过该阈值时提前停止；
                          None 表示不启用（CartPole-v1 通常以 475 视为“解决”）
    返回:
        训练好的 REINFORCEAgent
    """
    import gymnasium as gym

    # ---- 环境与随机种子 ----
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
    )

    recent_returns: List[float] = []  # 用于计算滑动平均回报

    for episode in range(1, num_episodes + 1):
        # 重置环境，得到初始状态
        state, _ = env.reset()

        log_probs: List[torch.Tensor] = []  # 收集每一步的 log π
        rewards: List[float] = []           # 收集每一步的奖励
        done = False

        # ---- 采样一条完整轨迹（一个 episode）----
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(float(reward))

            state = next_state
            done = terminated or truncated  # 终止或被截断都结束本 episode

        # ---- 用这条轨迹更新策略 ----
        agent.update(log_probs, rewards)

        # ---- 日志与提前停止 ----
        episode_return = sum(rewards)
        recent_returns.append(episode_return)
        if len(recent_returns) > 100:
            recent_returns.pop(0)
        avg_return = float(np.mean(recent_returns))

        if episode % log_interval == 0:
            print(
                f"Episode {episode:4d} | "
                f"本局回报: {episode_return:6.1f} | "
                f"近100局平均: {avg_return:6.1f}"
            )

        if solved_threshold is not None and len(recent_returns) >= 100 and avg_return >= solved_threshold:
            print(f"\n✅ 环境已解决！在第 {episode} 局达到平均回报 {avg_return:.1f}")
            break

    env.close()
    return agent


# ============================================================
# 4. 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE (Policy Gradient) 训练脚本")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="gymnasium 环境名")
    parser.add_argument("--episodes", type=int, default=600, help="训练 episode 数")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 γ")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--solved", type=float, default=475.0,
        help="提前停止的平均回报阈值（CartPole-v1 推荐 475）",
    )
    args = parser.parse_args()

    train(
        env_name=args.env,
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden,
        seed=args.seed,
        solved_threshold=args.solved,
    )
