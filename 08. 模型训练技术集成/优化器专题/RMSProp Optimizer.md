# RMSProp优化器

RMSProp（Root Mean Square Propagation）是一种常用的**自适应学习率优化器**，主要用于训练神经网络。它是对 Adagrad 优化器的改进，由 Geoffrey Hinton 在 2012 年提出。


## 背景

* **Adagrad 的问题**：Adagrad 会不断累积梯度平方和，导致学习率随时间单调下降，最终过小，影响训练。
* **RMSProp 的改进**：RMSProp 使用\*\*指数加权移动平均（Exponential Moving Average, EMA）\*\*来记录历史梯度平方，而不是简单累积，这样学习率不会无限缩小。


## 算法原理

对于每个参数 $\theta$，在第 $t$ 步更新时：

1. 计算梯度 $g_t$。
2. 更新梯度平方的移动平均：

<img width="290" height="45" alt="image" src="https://github.com/user-attachments/assets/88440188-10d0-4f98-bb9e-fede2630bfba" />


   其中 $\rho$ 通常取 0.9。
  
3. 使用调整后的学习率更新参数：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

   * $\eta$：全局学习率（通常在 0.001 左右）。
   * $\epsilon$：防止除零的常数（如 $1e-8$）。

---

## 特点

- 优点

* 可以自动调节不同参数的学习率，避免学习率过快衰减。
* 在非平稳目标（如 RNN 训练）中表现良好。
* 比 Adagrad 更稳定，收敛效果更好。

- 缺点

* 仍然需要人工选择初始学习率。
* 在某些任务中收敛速度可能不如 Adam。



## 应用场景

* 深度学习中的 RNN、LSTM 等模型训练。
* 图像、语音等任务的优化器选择之一。

---

## PyTorch 使用 RMSProp 的示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的前馈神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-08)

# 生成一些假数据
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()           # 梯度清零
    outputs = model(X)              # 前向传播
    loss = criterion(outputs, y)    # 计算损失
    loss.backward()                 # 反向传播
    optimizer.step()                # 更新参数

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
```

## 关键点说明

* `optim.RMSprop()` 就是 PyTorch 内置的 RMSProp 优化器。
* 重要参数：

  * `lr=0.01`：学习率（默认一般是 0.001）。
  * `alpha=0.9`：即算法中的 $\rho$，控制历史梯度平方的衰减率。
  * `eps=1e-08`：防止分母为 0 的小常数。



