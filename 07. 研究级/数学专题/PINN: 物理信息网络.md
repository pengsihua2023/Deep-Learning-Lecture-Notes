## PINN: 物理信息网络
### 原理和用法

#### **介绍**
物理信息神经网络（Physics-Informed Neural Networks, PINNs）是一种结合深度学习和物理规律的神经网络框架，用于求解偏微分方程（PDEs）或模拟物理系统。PINNs 将物理方程（如控制方程、初始条件、边界条件）嵌入神经网络的损失函数中，通过优化网络参数逼近 PDE 的解。相比传统数值方法（如有限差分、有限元），PINNs 无需离散化网格，适合高维或复杂几何问题。

#### **原理**
1. **核心思想**：
<img width="600" height="102" alt="image" src="https://github.com/user-attachments/assets/3229c1ce-599f-4f03-9153-0b8d649f7baf" />


* PINNs 使用神经网络 $u(x,t;\theta)$（参数为 $\theta$）逼近 PDE 的解 $u(x,t)$。
* 通过在损失函数中加入 PDE 残差、初始条件和边界条件，约束网络输出满足物理规律。
* 优化目标是最小化损失函数，使网络输出接近真实解。


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
<img width="627" height="89" alt="image" src="https://github.com/user-attachments/assets/6d6be84c-d5d6-4e07-8798-b1b4d2d761a7" />

- **损失函数**：
  - `compute_loss` 计算三部分损失：
<img width="828" height="141" alt="image" src="https://github.com/user-attachments/assets/1d70749d-14df-4136-ad55-d37a062de996" />

  - 总损失是三部分的加权和（这里权重设为 1，实际可调）。
- **数据**：
<img width="751" height="88" alt="image" src="https://github.com/user-attachments/assets/84132386-7ff5-44c3-8c0e-2c207cf76352" />

- **训练**：
  - 使用 Adam 优化器（更适合 PINNs，SGD 或 Adagrad 也可试）。
  - 训练 1000 次，打印损失。
- **测试**：
<img width="527" height="46" alt="image" src="https://github.com/user-attachments/assets/d3067031-2f34-48a7-b73e-5f5dff6c7dc1" />


---

#### **注意事项**
1. **超参数**：
<img width="661" height="129" alt="image" src="https://github.com/user-attachments/assets/c33a8fa0-368a-4796-ad97-afcedd7452f7" />

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
<img width="485" height="49" alt="image" src="https://github.com/user-attachments/assets/1048e018-8bfa-40fb-b18d-69bd36457dd8" />

     ```python
     import matplotlib.pyplot as plt
     plt.plot(x_test, u_pred.detach().numpy(), label='Predicted u(x, 0.5)')
     plt.xlabel('x'); plt.ylabel('u'); plt.legend(); plt.show()
     ```

---

#### **总结**
PINNs 是一种强大的方法，通过将物理方程嵌入神经网络损失函数，求解 PDE 或建模物理系统。PyTorch 实现简单，利用自动微分计算 PDE 残差，结合初始和边界条件优化网络。上述代码展示了 PINNs 在 Burgers 方程上的基本应用。

## 多维 PDE，反问题,及可视化
### 物理信息神经网络（PINN）更详细代码示例

基于之前的介绍和一维 Burgers 方程示例，以下提供更详细的 PINN 代码示例，涵盖：
- **多维 PDE**：以二维热传导方程为例（2D Laplace 方程的变体）。
- **反问题**：结合观测数据估计 PDE 中的未知参数（如扩散系数）。
- **可视化**：使用 Matplotlib 绘制预测解的热图和动画。

这些示例基于 PyTorch，代码更完整，包括数据生成、训练循环、损失记录和可视化。假设你有基本的 PyTorch 和 Matplotlib 环境。

#### 1. **多维 PDE 示例：二维热传导方程**
二维热传导方程（简化 Laplace 方程）：  
<img width="716" height="279" alt="image" src="https://github.com/user-attachments/assets/1502430f-1ab7-4586-b80c-3d10dfdf61ea" />


##### **代码示例**
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 定义神经网络
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),  # 输入 (x, y)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)  # 输出 u(x, y)
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

# 2. 定义损失函数
def compute_loss(model, x_f, y_f, x_b, y_b, u_b):
    x_f, y_f = x_f.requires_grad_(True), y_f.requires_grad_(True)  # 需要计算导数
    u = model(x_f, y_f)
    
    # PDE 残差：Laplace 方程 u_xx + u_yy = 0
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    pde_residual = u_xx + u_yy
    loss_pde = torch.mean(pde_residual ** 2)
    
    # 边界条件
    u_bc = model(x_b, y_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # 总损失
    return loss_pde + 10 * loss_bc  # 增加边界权重以加强约束

# 3. 准备数据
N_f = 10000  # PDE 内部采样点
N_b = 400    # 边界采样点 (每边 100 点)

# PDE 内部点
x_f = torch.rand(N_f, 1)
y_f = torch.rand(N_f, 1)

# 边界点
x_b_left = torch.zeros(N_b//4, 1)  # x=0
y_b_left = torch.rand(N_b//4, 1)
u_b_left = torch.zeros(N_b//4, 1)  # u=0

x_b_right = torch.ones(N_b//4, 1)  # x=1
y_b_right = torch.rand(N_b//4, 1)
u_b_right = torch.zeros(N_b//4, 1)  # u=0

x_b_bottom = torch.rand(N_b//4, 1)  # y=0
y_b_bottom = torch.zeros(N_b//4, 1)
u_b_bottom = torch.zeros(N_b//4, 1)  # u=0

x_b_top = torch.rand(N_b//4, 1)  # y=1
y_b_top = torch.ones(N_b//4, 1)
u_b_top = torch.sin(np.pi * x_b_top)  # u=sin(πx)

x_b = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)
u_b = torch.cat([u_b_left, u_b_right, u_b_bottom, u_b_top], dim=0)

# 4. 训练模型
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []  # 记录损失

for epoch in range(5000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, y_f, x_b, y_b, u_b)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. 可视化
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 绘制预测解热图
nx, ny = 100, 100
x_grid = torch.linspace(0, 1, nx).reshape(-1, 1)
y_grid = torch.linspace(0, 1, ny).reshape(-1, 1)
X, Y = torch.meshgrid(x_grid.squeeze(), y_grid.squeeze(), indexing='ij')
X_flat, Y_flat = X.reshape(-1, 1), Y.reshape(-1, 1)
u_pred = model(X_flat, Y_flat).detach().numpy().reshape(nx, ny)

plt.subplot(1, 2, 2)
plt.imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Solution')
plt.tight_layout()
plt.show()
```

##### **代码说明**
- **网络**：3 层全连接网络，使用 Tanh 激活，确保平滑。
- **损失**：PDE 残差（二阶导数）和边界条件损失，边界权重设为 10 以加强约束。
- **数据**：随机采样内部点和边界点，确保均匀分布。
- **训练**：5000 次迭代，记录损失曲线。
- **可视化**：损失曲线 + 预测解的 2D 热图，使用 `imshow` 显示温度分布。

#### 2. **反问题示例：参数估计**
<img width="951" height="47" alt="image" src="https://github.com/user-attachments/assets/7d2f8d41-8fe8-4c8f-a250-f4f30cf79cf0" />


##### **代码示例**
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义神经网络（添加可学习参数 nu）
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
        self.nu = nn.Parameter(torch.tensor(0.1))  # 初始猜测 nu，可学习

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. 定义损失函数（添加观测损失）
def compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, x_o, t_o, u_o):
    x_f, t_f = x_f.requires_grad_(True), t_f.requires_grad_(True)
    u = model(x_f, t_f)
    
    # PDE 残差
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t + u * u_x - model.nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # 初始条件
    u_init = model(x_i, t_i)
    loss_init = torch.mean((u_init - u_i) ** 2)
    
    # 边界条件
    u_bc = model(x_b, t_b)
    loss_bc = torch.mean((u_bc - u_b) ** 2)
    
    # 观测数据损失
    u_obs = model(x_o, t_o)
    loss_obs = torch.mean((u_obs - u_o) ** 2)
    
    # 总损失
    return loss_pde + loss_init + loss_bc + loss_obs

# 3. 准备数据（添加观测数据）
N_f, N_i, N_b, N_o = 2000, 100, 100, 50  # 添加观测点
true_nu = 0.01 / np.pi  # 真实 nu

# PDE、初始、边界数据（同之前一维示例）
x_f = torch.rand(N_f, 1) * 2 - 1
t_f = torch.rand(N_f, 1)
x_i = torch.rand(N_i, 1) * 2 - 1
t_i = torch.zeros(N_i, 1)
u_i = -torch.sin(np.pi * x_i)
x_b = torch.cat([torch.ones(N_b//2, 1), -torch.ones(N_b//2, 1)])
t_b = torch.rand(N_b, 1)
u_b = torch.zeros(N_b, 1)

# 观测数据（模拟：使用真实解 + 噪声）
x_o = torch.rand(N_o, 1) * 2 - 1
t_o = torch.rand(N_o, 1)
# 模拟观测 u_o（假设有解析解或数值解，这里简化用 sin 近似 + 噪声）
u_o = -torch.sin(np.pi * x_o) * torch.exp(-true_nu * np.pi**2 * t_o) + 0.01 * torch.randn(N_o, 1)

# 4. 训练模型
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
nu_history = []  # 记录 nu 估计

for epoch in range(2000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b, x_o, t_o, u_o)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    nu_history.append(model.nu.item())
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Estimated nu: {model.nu.item():.6f}")

# 5. 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(nu_history)
plt.axhline(true_nu, color='r', linestyle='--', label='True nu')
plt.xlabel('Epoch')
plt.ylabel('nu')
plt.title('nu Estimation')
plt.legend()

# 动画可视化：u(x, t) 随时间变化
fig, ax = plt.subplots()
x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
def animate(t_frame):
    ax.clear()
    t_test = torch.ones_like(x_test) * (t_frame / 50.0)  # t from 0 to 1
    u_pred = model(x_test, t_test).detach().numpy()
    ax.plot(x_test.numpy(), u_pred, label=f't={t_frame/50:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
ani = FuncAnimation(fig, animate, frames=50, interval=100)
plt.close(fig)  # 防止静态显示
ani.save('burgers_animation.gif', writer='imagemagick')  # 保存为 GIF（需安装 imagemagick）
plt.show()  # 或直接显示动画
```

##### **代码说明**
- **网络**：添加 `self.nu` 作为可学习参数（初始 0.1）。
- **损失**：新增观测损失 `loss_obs`，使用模拟观测数据（真实解 + 噪声）。
- **训练**：记录 `nu_history`，估计 nu。
- **可视化**：损失曲线、nu 估计曲线 + u(x, t) 的动画（保存为 GIF）。

#### 3. **注意事项和扩展**
- **多维 PDE**：对于 3D 或更高维，输入层改为 `nn.Linear(3, ...)` 或更多，采样点需均匀分布。
- **反问题**：观测数据可从实验或模拟获取；对于更复杂参数（如函数参数），可使用额外网络表示。
- **可视化**：动画使用 `FuncAnimation`，适合动态 PDE；热图适合稳态问题。
- **优化**：若收敛慢，可用 L-BFGS 优化器或增加采样点。
- **运行**：代码需 PyTorch、NumPy、Matplotlib。动画需 imagemagick 支持 GIF 输出。

