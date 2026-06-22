# Handwritten AI ✍️

> 从零开始手写实现经典 AI 算法 —— 涵盖**机器学习**、**深度学习**与**强化学习**三大板块。
> Implementations of classic Machine Learning / Deep Learning / Reinforcement Learning algorithms from scratch, for educational purposes.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 📖 简介 (Introduction)

本项目 **handwritten-ai** 旨在通过**手写代码**（尽量减少对高层 API 的依赖）来深入理解人工智能领域的核心算法。

与其只调用 `sklearn.linear_model.LogisticRegression` 或 `torch.nn.Transformer`，不如亲自把背后的数学推导和迭代过程实现一遍。这个仓库适合：

* 想要深入理解算法底层原理的学习者；
* 需要复习经典模型细节的研究者 / 求职面试者；
* 希望对照公式逐行阅读实现的同学。

每个算法都力求做到：**数学原理注释清晰**、**代码可独立运行**、**附带最简使用示例**。

## 🗂️ 项目结构 (Project Structure)

```
handwritten-ai/
├── machine_learning/        # 机器学习：经典统计学习算法
├── deep_learning/           # 深度学习：神经网络与现代架构
│   └── transformer/         # ✅ Transformer 已实现（encoder-decoder）
└── reinforcement_learning/  # 强化学习：智能体与决策算法
    └── policy_gradient/     # ✅ REINFORCE 已实现
```

> 说明：标记 ✅ 的为已实现并验证；🚧 为骨架已搭建、实现中；其余为规划中的路线图 (roadmap)。

## 🧩 算法路线图 (Roadmap)

### 📊 机器学习 (Machine Learning)

> 经典统计学习算法，主要依赖 NumPy 手写实现。

* [ ] **线性回归 (Linear Regression)** —— 最小二乘 / 梯度下降
* [ ] **逻辑回归 (Logistic Regression)**
* [ ] **K 近邻 (k-Nearest Neighbors, kNN)**
* [ ] **朴素贝叶斯 (Naive Bayes)**
* [ ] **决策树 (Decision Tree, ID3 / CART)**
* [ ] **支持向量机 (Support Vector Machine, SVM)**
* [ ] **K-Means 聚类**
* [ ] **主成分分析 (PCA)**
* [ ] **集成学习 (Bagging / Random Forest / AdaBoost / GBDT)**

### 🧠 深度学习 (Deep Learning)

> 神经网络基础组件与现代架构，基于 PyTorch 张量与自动求导手写。

* [ ] **多层感知机 (MLP)** + 反向传播 (Backpropagation)
* [ ] **卷积神经网络 (CNN)**
* [ ] **循环神经网络 (RNN / LSTM / GRU)**
* [x] **Transformer** → [`deep_learning/transformer`](deep_learning/transformer)（已实现并验证）
    * Self-Attention Mechanism
    * Multi-Head Attention
    * Positional Encoding
    * Encoder & Decoder Layers

### 🎮 强化学习 (Reinforcement Learning)

> 从价值方法到策略方法，逐步构建完整的 RL 算法体系。

* [ ] **Q-Learning / DQN** —— 基于价值 (Value-based)
* [x] **Policy Gradient — REINFORCE** —— 基于策略 (Policy-based) → [`reinforcement_learning/policy_gradient`](reinforcement_learning/policy_gradient)
* [ ] **Actor-Critic / A2C / A3C**
* [ ] **PPO (Proximal Policy Optimization)**

## 🚀 快速开始 (Quick Start)

```bash
# 克隆仓库
git clone <repo-url>
cd handwritten-ai

# 安装依赖（按需）
pip install numpy torch gymnasium

# 以已实现的 REINFORCE 为例：训练 + 评估
python reinforcement_learning/policy_gradient/example.py
```

各算法的详细说明与运行方式见对应子目录下的 `README.md`。

## 🛠️ 环境依赖 (Requirements)

* Python 3.8+
* NumPy（机器学习部分）
* PyTorch（深度学习 / 强化学习部分）
* Gymnasium（强化学习环境）

## 📜 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源。
