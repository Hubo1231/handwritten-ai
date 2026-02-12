# Handwritten AI ✍️

> 从零开始手写实现经典 AI 算法（Transformer, Reinforcement Learning 等）。
> Implementations of classic AI algorithms from scratch for educational purposes.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 📖 简介 (Introduction)

本项目 **handwritten-ai** 旨在通过**手写代码**（尽量减少对高层 API 的依赖）来深入理解人工智能领域的核心算法。

与其只调用 `torch.nn.Transformer`，不如亲自实现一遍 `Multi-Head Attention`。这个仓库适合：
* 想要深入理解算法底层原理的学习者。
* 需要复习经典模型细节的研究者/面试者。

## 🧩 算法列表 (Implemented Algorithms)

### 🤖 深度学习 (Deep Learning)
* [x] **Transformer**
    * Self-Attention Mechanism
    * Multi-Head Attention
    * Positional Encoding
    * Encoder & Decoder Layers
* [ ] **CNN Variants** (ResNet, etc.)

### 🎮 强化学习 (Reinforcement Learning)
* [x] **Q-Learning / DQN**
* [ ] **Policy Gradient**
* [ ] **PPO (Proximal Policy Optimization)**
* [ ] **A3C**

### 📊 经典机器学习 (Classic ML)
* [ ] K-Means
* [ ] SVM
* [ ] Decision Tree

## 🚀 快速开始 (Quick Start)

### 环境依赖
```bash
git clone [https://github.com/your-username/handwritten-ai.git](https://github.com/your-username/handwritten-ai.git)
cd handwritten-ai
pip install -r requirements.txt
