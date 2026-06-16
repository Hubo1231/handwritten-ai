"""
REINFORCE 最简使用示例
======================

本文件演示如何用几行代码调用 reinforce.py 中实现的 REINFORCE 算法，
在经典控制任务 CartPole-v1（倒立摆）上训练一个智能体，并在训练完成后
用学到的策略跑几局来直观地看看效果。

运行方式（在仓库根目录或本目录下）：

    python reinforcement_learning/policy_gradient/example.py
"""

import numpy as np
import torch

# 注意：如果直接 `python example.py` 运行，下面这种相对/同目录导入即可生效；
# 因为 example.py 与 reinforce.py 在同一个文件夹中。
from reinforce import REINFORCEAgent, train


def evaluate(agent: REINFORCEAgent, env_name: str = "CartPole-v1", episodes: int = 5) -> None:
    """用训练好的（贪心）策略评估几局，打印每局回报。

    评估时我们选择概率最大的动作（argmax），而非随机采样，
    以观察策略“最优表现”而不受探索噪声影响。
    """
    import gymnasium as gym

    env = gym.make(env_name)
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # 直接取 logits 最大的动作（确定性 / 贪心策略）
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = agent.policy(state_tensor)
            action = int(torch.argmax(logits, dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"[评估] 第 {ep} 局回报: {total_reward:.1f}")
    env.close()


def main() -> None:
    # ---- 1. 训练 ----
    # 直接调用封装好的 train() 即可。这里把 episode 数调小一点，方便快速看到结果。
    print("开始训练 REINFORCE 智能体（CartPole-v1）...\n")
    agent = train(
        env_name="CartPole-v1",
        num_episodes=600,
        gamma=0.99,
        lr=1e-2,
        seed=0,
        log_interval=20,
        solved_threshold=475.0,  # 平均回报达到 475 即提前停止
    )

    # ---- 2. 评估 ----
    print("\n训练结束，用学到的策略评估 5 局：")
    evaluate(agent, env_name="CartPole-v1", episodes=5)


if __name__ == "__main__":
    main()
