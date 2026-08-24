"""用纯 Python 手写一个最小的反向传播例子。

网络结构：

    x --(乘以 w)--> h --(乘以 v)--> y_hat --> loss

对应公式：

    h     = w * x
    y_hat = v * h
    loss  = 0.5 * (y_hat - y) ** 2

运行方式：

    python deep_learning/backpropagation/example.py
"""

from typing import Tuple


def forward(x: float, y: float, w: float, v: float) -> Tuple[float, float, float]:
    """前向传播：用当前参数计算隐藏值、预测值和损失。"""
    h = w * x
    y_hat = v * h
    loss = 0.5 * (y_hat - y) ** 2
    return h, y_hat, loss


def backward(x: float, y: float, w: float, v: float) -> Tuple[float, float]:
    """反向传播：根据链式法则，手算 loss 对 w、v 的梯度。"""
    h, y_hat, _ = forward(x, y, w, v)

    # 从 loss 开始，沿计算图从右向左传播。
    dloss_dy_hat = y_hat - y       # dloss / dy_hat

    # v -> y_hat -> loss
    dy_hat_dv = h                  # dy_hat / dv
    dloss_dv = dloss_dy_hat * dy_hat_dv

    # w -> h -> y_hat -> loss
    dy_hat_dh = v                  # dy_hat / dh
    dh_dw = x                      # dh / dw
    dloss_dw = dloss_dy_hat * dy_hat_dh * dh_dw

    return dloss_dw, dloss_dv


def numerical_gradient(
    x: float,
    y: float,
    w: float,
    v: float,
    parameter: str,
    epsilon: float = 1e-5,
) -> float:
    """用微小扰动近似梯度，用来检查手算结果。"""
    if parameter == "w":
        loss_plus = forward(x, y, w + epsilon, v)[2]
        loss_minus = forward(x, y, w - epsilon, v)[2]
    elif parameter == "v":
        loss_plus = forward(x, y, w, v + epsilon)[2]
        loss_minus = forward(x, y, w, v - epsilon)[2]
    else:
        raise ValueError("parameter 必须是 'w' 或 'v'")

    return (loss_plus - loss_minus) / (2 * epsilon)


def main() -> None:
    # 一条训练数据和两个待学习参数。
    x = 3.0
    y = 1.0
    w = 0.5
    v = 2.0
    learning_rate = 0.01

    # 1. 前向传播
    h, y_hat, loss = forward(x, y, w, v)
    print("========== 1. 前向传播 ==========")
    print(f"h     = w * x              = {w} * {x} = {h}")
    print(f"y_hat = v * h              = {v} * {h} = {y_hat}")
    print(f"loss  = 0.5 * (y_hat-y)^2  = {loss}\n")

    # 2. 反向传播
    dloss_dw, dloss_dv = backward(x, y, w, v)
    print("========== 2. 反向传播 ==========")
    print(f"d(loss)/d(y_hat) = y_hat - y = {y_hat} - {y} = {y_hat - y}")
    print(
        "d(loss)/d(v)     = d(loss)/d(y_hat) * d(y_hat)/d(v)\n"
        f"                 = {y_hat - y} * {h} = {dloss_dv}"
    )
    print(
        "d(loss)/d(w)     = d(loss)/d(y_hat) * d(y_hat)/d(h) * d(h)/d(w)\n"
        f"                 = {y_hat - y} * {v} * {x} = {dloss_dw}"
    )
    print(f"所以梯度向量为 (dL/dw, dL/dv) = ({dloss_dw}, {dloss_dv})\n")

    # 3. 梯度含义：参数增加一点，损失大约增加“梯度 * 变化量”。
    delta = 0.001
    loss_after_small_w_change = forward(x, y, w + delta, v)[2]
    predicted_loss_change = dloss_dw * delta
    actual_loss_change = loss_after_small_w_change - loss
    print("========== 3. 梯度的直观含义 ==========")
    print(f"让 w 增加一个很小的量 {delta}：")
    print(f"梯度预测的损失变化 = dL/dw * {delta} = {predicted_loss_change:.6f}")
    print(f"实际损失变化       = {actual_loss_change:.6f}")
    print("二者很接近，这就是梯度所描述的局部变化速度。\n")

    # 用数值微分再次验证手算梯度。
    numerical_dw = numerical_gradient(x, y, w, v, "w")
    numerical_dv = numerical_gradient(x, y, w, v, "v")
    print("========== 4. 数值微分校验 ==========")
    print(f"w：手算梯度 = {dloss_dw:.6f}，数值梯度 = {numerical_dw:.6f}")
    print(f"v：手算梯度 = {dloss_dv:.6f}，数值梯度 = {numerical_dv:.6f}\n")

    # 5. 梯度下降：梯度指向损失上升最快的方向，所以往反方向更新。
    new_w = w - learning_rate * dloss_dw
    new_v = v - learning_rate * dloss_dv
    _, new_y_hat, new_loss = forward(x, y, new_w, new_v)
    print("========== 5. 梯度下降更新 ==========")
    print(f"w_new = w - lr * dL/dw = {w} - {learning_rate} * {dloss_dw} = {new_w}")
    print(f"v_new = v - lr * dL/dv = {v} - {learning_rate} * {dloss_dv} = {new_v}")
    print(f"更新后的预测值：{new_y_hat:.6f}")
    print(f"更新前的损失：{loss:.6f}")
    print(f"更新后的损失：{new_loss:.6f}")
    print("损失下降了，说明参数更新方向正确。")


if __name__ == "__main__":
    main()
