## Deep Backward Stochastic Differential Equation (Deep BSDE) Method
Deep Backward Stochastic Differential Equation (Deep BSDE) Method 是一种基于深度学习的数值方法，用于求解高维（甚至数百维）非线性抛物型偏微分方程（PDE），特别是在传统网格方法（如有限差分）因维度灾难而失效的场景中。它由 Jiequn Han、Arnulf Jentzen 和 Weinan E 于 2017 年提出，将 PDE 通过 Feynman-Kac 定理转化为后向随机微分方程（BSDE），然后使用神经网络逼近 BSDE 的解和梯度，从而通过最小化终端条件的损失来训练模型。Deep BSDE 具有网格无关性（mesh-free），适用于金融定价（如高维 Black-Scholes）、量子力学和控制问题，但计算成本较高，且收敛性依赖于时间离散化和网络架构。

与 Deep Galerkin Method (DGM) 或 Deep Ritz Method (DRM) 相比，Deep BSDE 更适合时间相关的抛物型 PDE，并自然处理随机性，但需要模拟随机路径，可能引入方差。

### 数学描述
<img width="1010" height="922" alt="image" src="https://github.com/user-attachments/assets/9c3ef52c-b02b-447d-b2ae-d460053c66cf" />


### 代码实现
<img width="888" height="115" alt="image" src="https://github.com/user-attachments/assets/8f855fa3-dac5-411d-8a32-6d289a76d75e" />


```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 神经网络逼近 Z_t (梯度过程)，对于每个时间步共享或独立
class ZNet(nn.Module):
    def __init__(self, d=1, hidden_size=32):  # d 是维度，这里1D
        super(ZNet, self).__init__()
        self.fc1 = nn.Linear(d + 1, hidden_size)  # 输入 (t, x)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d)   # 输出 Z \in \mathbb{R}^d
    
    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)

# Deep BSDE 求解器
class DeepBSDE:
    def __init__(self, d=1, T=1.0, N=20, batch_size=256, lr=0.01):
        self.d = d  # 维度
        self.T = T  # 终端时间
        self.N = N  # 时间步数
        self.dt = T / N
        self.batch_size = batch_size
        self.y0 = nn.Parameter(torch.tensor([0.5]))  # 初始 Y0 (u(0,0))
        self.z_net = ZNet(d=d)  # 共享 Z 网络
        self.optimizer = torch.optim.Adam(list(self.z_net.parameters()) + [self.y0], lr=lr)
    
    def f(self, y):  # 非线性项 f(y) = y^3
        return y ** 3
    
    def g(self, x):  # 终端条件 g(x) = cos(pi x / 2)
        return torch.cos(np.pi * x / 2)
    
    def simulate_paths(self, x0=0.0):
        # 模拟 Brownian 路径
        dw = torch.sqrt(torch.tensor(self.dt)) * torch.randn((self.batch_size, self.N, self.d))
        x = torch.zeros((self.batch_size, self.N+1, self.d))
        x[:, 0, :] = x0
        for n in range(self.N):
            x[:, n+1, :] = x[:, n, :] + dw[:, n, :]  # sigma=1, mu=0
        return x, dw
    
    def loss(self, x, dw):
        y = self.y0.repeat(self.batch_size, 1)  # Y_0
        t = torch.zeros((self.batch_size, 1))
        for n in range(self.N):
            z = self.z_net(t, x[:, n, :])  # Z_n
            y = y - self.f(y) * self.dt + torch.sum(z * dw[:, n, :], dim=1, keepdim=True)
            t = t + self.dt
        return torch.mean((y - self.g(x[:, -1, :])) ** 2)
    
    def train(self, epochs=2000, x0=0.0):
        losses = []
        for epoch in range(epochs):
            x, dw = self.simulate_paths(x0)
            loss_val = self.loss(x, dw)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            losses.append(loss_val.item())
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val.item():.4f}, y0: {self.y0.item():.4f}")
        return losses

# 主程序
dbsde = DeepBSDE(d=1, T=1.0, N=50, batch_size=512, lr=0.005)
losses = dbsde.train()

# 测试：估计 u(0,0)
print(f"Estimated u(0,0): {dbsde.y0.item()}")

# 可视化损失
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 为了可视化 u(t,x)，需修改为评估多个点，但这里焦点在 y0
```

### 代码解释
<img width="863" height="305" alt="image" src="https://github.com/user-attachments/assets/c460de85-baf3-47dc-9ce0-796b5e0286f9" />
