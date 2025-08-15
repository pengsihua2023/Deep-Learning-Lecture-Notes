## Deep Galerkin Method (DGM)
Deep Galerkin Method (DGM)是一种深度学习算法，用于求解偏微分方程（PDE），特别是在高维情况下。它由Justin Sirignano和Konstantinos Spiliopoulos于2017年提出，灵感来源于经典的Galerkin方法，但用深度神经网络取代了传统的有限维基函数来逼近PDE的解。DGM通过最小化PDE残差的积分形式来训练网络，避免了网格划分，具有网格无关性（mesh-free），适用于非线性、高维和复杂域的PDE求解，如金融中的Black-Scholes方程或流体力学问题。

与Physics-Informed Neural Networks (PINN)类似，DGM也使用神经网络表示解，但DGM更强调Galerkin正交条件，通过随机采样积分域来近似损失函数，从而高效处理高维问题。它已被扩展到各种PDE，如Fokker-Planck方程、Stokes方程和均场游戏。

![Uploading image.png…]()


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 神经网络逼近u(x)
class DGMNet(nn.Module):
    def __init__(self, hidden_size=50, num_layers=3):
        super(DGMNet, self).__init__()
        self.fc_in = nn.Linear(1, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.fc_out(x)

# PDE残差：-u''(x) - pi^2 sin(pi x)
def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = (np.pi ** 2) * torch.sin(np.pi * x)
    return -u_xx - f  # 残差

# 边界损失：u(0)=0, u(1)=0
def boundary_loss(model):
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    return model(x0)**2 + model(x1)**2

# 训练
def train(model, optimizer, num_epochs=5000, num_samples=100, lambda_bd=1.0):
    losses = []
    for epoch in range(num_epochs):
        # 内部采样
        x_int = torch.rand((num_samples, 1))
        res = pde_residual(model, x_int)
        loss_int = torch.mean(res ** 2)
        loss_bd = boundary_loss(model)
        loss = loss_int + lambda_bd * loss_bd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return losses

# 主程序
model = DGMNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses = train(model)

# 测试
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
u_pred = model(x_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy())
plt.plot(x_test.numpy(), u_pred, label='Prediction')
plt.plot(x_test.numpy(), u_true, label='True')
plt.legend()
plt.show()
```

### 代码解释
1. **DGMNet**：一个简单的全连接网络，使用tanh激活，输入1维（x），输出1维（u(x)）。
2. **pde_residual**：使用自动微分计算二阶导数，并计算PDE残差。
3. **boundary_loss**：直接强制边界条件（硬约束，也可软约束）。
4. **train**：每轮随机采样内部点，计算损失（内部残差 + 边界损失），优化网络。运行后，损失下降，预测曲线接近真实解。

这个实现是最简单的演示；实际中可扩展到高维PDE或添加自适应采样。 
