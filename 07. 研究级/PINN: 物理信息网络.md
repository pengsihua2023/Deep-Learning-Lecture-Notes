## PINN: 物理信息网络
### 原理和用法

#### **介绍**
物理信息神经网络（Physics-Informed Neural Networks, PINNs）是一种结合深度学习和物理规律的神经网络框架，用于求解偏微分方程（PDEs）或模拟物理系统。PINNs 将物理方程（如控制方程、初始条件、边界条件）嵌入神经网络的损失函数中，通过优化网络参数逼近 PDE 的解。相比传统数值方法（如有限差分、有限元），PINNs 无需离散化网格，适合高维或复杂几何问题。

#### **原理**
1. **核心思想**：
<img width="600" height="102" alt="image" src="https://github.com/user-attachments/assets/3229c1ce-599f-4f03-9153-0b8d649f7baf" />


2. **PDE 形式**：考虑一般形式的 PDE：   
<img width="517" height="258" alt="image" src="https://github.com/user-attachments/assets/530507c3-2e1a-4779-8c11-44bdc3a3a1f1" />

3. **损失函数**：
   PINNs 的损失函数由三部分组成：
<img width="562" height="477" alt="image" src="https://github.com/user-attachments/assets/113895c6-df38-4dd9-afdf-c9f434d16f16" />


4. **自动微分**：
<img width="704" height="70" alt="image" src="https://github.com/user-attachments/assets/38497436-1a0e-4636-a0f8-7eee56bf6d0f" />


5. **优点**：
   - 无需网格划分，适合高维或复杂几何。
   - 灵活，易于融入物理约束。
   - 可结合少量观测数据和物理规律。

6. **缺点**：
<img width="532" height="137" alt="image" src="https://github.com/user-attachments/assets/4531f7e6-3c8a-4535-9c3e-466ac5b9a795" />


7. **适用场景**：
   - 求解 PDE（如热传导、波动方程、流体力学）。
   - 数据驱动的物理建模（如结合实验数据）。
   - 反问题（如参数估计）。

---

#### **PyTorch 用法**
以下是最简洁的 PyTorch 代码示例，展示如何使用 PINNs 求解一维 Burgers 方程（一个非线性 PDE），并说明其原理和使用方法。

##### **问题描述**
<img width="758" height="382" alt="image" src="https://github.com/user-attachments/assets/6399f215-6598-43db-b06f-274c0a1dbdce" />


##### **代码示例**
```python
import torch
import torch.nn as nn
import numpy as np

# 1. 定义神经网络
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # 输入 (x, t)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)  # 输出 u(x, t)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. 定义损失函数
def compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, nu=0.01/np.pi):
    x_f, t_f = x_f.requires_grad_(True), t_f.requires_grad_(True)  # 需要计算导数
    u = model(x_f, t_f)
    
    # PDE 残差
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # 初始条件
    u_init = model(x_i, t_i)
    loss_init = torch.mean((u_init - u_i) ** 2)
    
    # 边界条件
    u_bc = model(x_b, t_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # 总损失
    return loss_pde + loss_init + loss_bc

# 3. 准备数据
N_f, N_i, N_b = 2000, 100, 100
x_f = torch.rand(N_f, 1) * 2 - 1  # x in [-1, 1]
t_f = torch.rand(N_f, 1)  # t in [0, 1]
x_i = torch.rand(N_i, 1) * 2 - 1  # 初始条件 x
t_i = torch.zeros(N_i, 1)  # t = 0
u_i = -torch.sin(np.pi * x_i)  # u(x, 0) = -sin(πx)
x_b = torch.cat([torch.ones(N_b//2, 1), -torch.ones(N_b//2, 1)])  # x = ±1
t_b = torch.rand(N_b, 1)  # t in [0, 1]
u_b = torch.zeros(N_b, 1)  # u(±1, t) = 0

# 4. 训练模型
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 5. 测试预测
x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
t_test = torch.ones(100, 1) * 0.5  # t = 0.5
u_pred = model(x_test, t_test)
print("预测结果形状:", u_pred.shape)  # 输出：torch.Size([100, 1])
```

##### **代码说明**
- **模型**：
  - 定义一个简单全连接神经网络，输入为 `(x, t)`，输出为 \( u(x, t) \)。
  - 使用 `Tanh` 激活函数，确保输出平滑，适合 PDE 解。
- **损失函数**：
  - `compute_loss` 计算三部分损失：
    - **PDE 残差**：使用 `torch.autograd.grad` 计算 \( u_t, u_x, u_xx \)，构造 Burgers 方程残差。
    - **初始条件**：确保 \( u(x, 0) \approx -\sin(\pi x) \)。
    - **边界条件**：确保 \( u(-1, t) = u(1, t) = 0 \)。
  - 总损失是三部分的加权和（这里权重设为 1，实际可调）。
- **数据**：
  - 随机采样 PDE 点（`x_f, t_f`）、初始点（`x_i, t_i`）和边界点（`x_b, t_b`）。
  - 初始条件 \( u_i = -\sin(\pi x) \)，边界条件 \( u_b = 0 \)。
- **训练**：
  - 使用 Adam 优化器（更适合 PINNs，SGD 或 Adagrad 也可试）。
  - 训练 1000 次，打印损失。
- **测试**：
  - 在 \( t = 0.5, x \in [-1, 1] \) 上预测解，验证输出形状。

---

#### **注意事项**
1. **超参数**：
   - 采样点数（`N_f, N_i, N_b`）：增加点数提高精度，但增加计算成本。
   - 网络结构：层数和神经元数需根据 PDE 复杂性调整。
   - 损失权重：复杂 PDE 可能需调整 \( \lambda_1, \lambda_2, \lambda_3 \)。
2. **自动微分**：
   - 使用 `requires_grad_(True)` 确保计算高阶导数。
   - 确保输入张量格式正确（形状为 `[N, 1]`）。
3. **优化器**：
   - Adam 比 SGD 或 Adagrad 更适合 PINNs，因其自适应性。
   - 可尝试 L-BFGS 优化器以提高收敛性。
4. **实际应用**：
   - 替换 Burgers 方程为其他 PDE（如热方程、波动方程）。
   - 加入观测数据，解决反问题。
   - 使用 GPU 加速训练：`model.cuda(), x_f.cuda(), ...`。
5. **可视化**：
   - 可添加 matplotlib 代码绘制 \( u(x, t) \) 的预测结果：
     ```python
     import matplotlib.pyplot as plt
     plt.plot(x_test, u_pred.detach().numpy(), label='Predicted u(x, 0.5)')
     plt.xlabel('x'); plt.ylabel('u'); plt.legend(); plt.show()
     ```

---

#### **总结**
PINNs 是一种强大的方法，通过将物理方程嵌入神经网络损失函数，求解 PDE 或建模物理系统。PyTorch 实现简单，利用自动微分计算 PDE 残差，结合初始和边界条件优化网络。上述代码展示了 PINNs 在 Burgers 方程上的基本应用。
