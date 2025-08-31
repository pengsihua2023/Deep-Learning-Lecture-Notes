## 神经算子(Neural Operators)
神经算子（Neural Operators）是一种特殊的神经网络架构，用于学习从一个函数空间到另一个函数空间的映射（即学习“算子”）。传统的神经网络通常处理点到点的映射（如图像分类中的像素到标签），而神经算子可以处理函数到函数的映射，例如在科学计算中快速近似求解偏微分方程（PDE），它具有网格无关性（resolution-invariant），意味着训练时用一种分辨率的数据，推理时可以用不同的分辨率。

### 数学描述
<img width="982" height="671" alt="image" src="https://github.com/user-attachments/assets/ff261a8f-9397-4c75-a14e-f93170346591" />


神经算子旨在近似一个算子 $G : \mathcal{A} \to \mathcal{U}\$，其中 $\mathcal{A}\$ 和 \$\mathcal{U}\$ 是 Banach 空间（通常是函数空间，如 $L^2(D)\$），\$D \subset \mathbb{R}^d\$ 是域。给定输入函数 $a \in \mathcal{A}\$ ，目标是预测输出函数 $u = G(a) \in \mathcal{U}\$ 。

神经算子 $G^\theta\$ （参数为 $\theta\$ ）被设计为网格无关的，即它不依赖于输入函数的离散化分辨率。形式上，它可以表示为层级结构：

$$
v_0(x) = P(a(x)), \quad v_{t+1}(x) = \sigma \big( W v_t(x) + (\mathcal{K}(a;\phi) v_t)(x) \big), \quad t = 0, \ldots, T-1,
$$

$$
G^\theta(a)(x) = Q(v_T(x)),
$$

其中：

* $P : \mathbb{R}^{d\_a} \to \mathbb{R}^{d\_v}\$ 是提升映射（lifting），将输入维度从 $d\_a\$ 提升到更高维度 \$d\_v\$ 。
* $Q : \mathbb{R}^{d\_v} \to \mathbb{R}^{d\_u}\$ 是投影映射（projection），将隐藏维度投影到输出维度 $d\_u\$ 。
* $W\$ 是局部线性变换（点到点）。
* $\mathcal{K}\$ 是非局部积分核算子：

$$
(\mathcal{K}(a;\phi)v)(x) = \int_D \kappa_\phi(x,y,a(x),a(y)) v(y)\, dy,
$$

其中 \$\kappa\_\phi\$ 是由神经网络参数化的核函数。

* $\sigma\$ 是激活函数（如 GELU）。



<img width="1021" height="523" alt="image" src="https://github.com/user-attachments/assets/8f275f27-a9f9-4b14-b635-ebcd5720c1c2" />


一个经典的实现是傅里叶神经算子（Fourier Neural Operator, FNO），它利用傅里叶变换在频域中高效参数化
 $\mathcal{K}\$ ，避免直接计算积分。假设周期边界条件，在1D情况下：

$$
(\mathcal{K}v)(x) = \mathcal{F}^{-1} \Big( R_\phi \cdot (\mathcal{F}v) \Big)(x),
$$

其中：

* $\mathcal{F}\$ 是傅里叶变换： $(\mathcal{F}v)_k = \int_D v(x)e^{-2\pi i k \cdot x} dx \quad (\text{离散时用 FFT})$

* $\mathcal{F}^{-1}\$ 是逆傅里叶变换。

* $R\_\phi\$ 是可学习参数矩阵（复数），截断到前 $k\_{\max}\$ 个低频模式：对于每个模式 $k\$,
  $R\_\phi(k) \in \mathbb{C}^{d\_v \times d\_v}\$。这使得 \$\mathcal{K}\$ 成为全局卷积，计算复杂度为 \$O(N \log N)\$，其中 \$N\$ 是网格点数。


FNO 的优势在于分辨率无关性：训练时用粗网格，推理时可用于细网格，因为傅里叶模式是连续的。

下面是用 PyTorch 从零实现的一个最简单的 1D FNO 例子。我们假设任务是学习一个简单的算子：将输入函数
$f(x) = \sin(kx)\$ 映射到其积分形式（累积积分）。代码包括谱卷积层（SpectralConv1d）和 FNO 模型，生成随机数据进行训练演示。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 谱卷积层（核心组件，使用FFT在频域乘以权重）
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 傅里叶模式的个数
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x shape: (batch, in_channels, grid_size)
        x_ft = torch.fft.rfft(x)  # 实数FFT
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # 逆FFT
        return x

# 简单的1D FNO模型
class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64):
        super(FNO1d, self).__init__()
        self.conv0 = SpectralConv1d(1, width, modes)  # 输入通道1
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(1, width, 1)  # 线性层
        self.w1 = nn.Conv1d(width, width, 1)
        self.fc = nn.Linear(width, 1)  # 输出层

    def forward(self, x):
        # x shape: (batch, grid_size, 1) -> 转置为 (batch, 1, grid_size)
        x = x.permute(0, 2, 1)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = x.permute(0, 2, 1)  # 转回 (batch, grid_size, width)
        x = self.fc(x)
        return x.squeeze(-1)  # 输出 (batch, grid_size)

# 生成简单数据：输入f(x) = sin(k x)，目标g(x) = -cos(k x)/k (积分从0到x，忽略常数)
grid_size = 256
n_train = 1000
x = torch.linspace(0, 2 * np.pi, grid_size).unsqueeze(0).repeat(n_train, 1)  # (n_train, grid_size)
k = torch.randint(1, 5, (n_train, 1))  # 随机频率
input = torch.sin(k * x)  # (n_train, grid_size)
target = -torch.cos(k * x) / k  # 积分结果 (n_train, grid_size)
input = input.unsqueeze(-1)  # (n_train, grid_size, 1)
target = target.unsqueeze(-1)

# 训练模型
model = FNO1d()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):  # 简单训练100轮
    optimizer.zero_grad()
    out = model(input)
    loss = F.mse_loss(out.unsqueeze(-1), target)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 测试
test_input = torch.sin(3 * x[0]).unsqueeze(0).unsqueeze(-1)  # 新样本 k=3
pred = model(test_input).detach().numpy()
true = -np.cos(3 * x[0].numpy()) / 3
plt.plot(x[0].numpy(), pred[0], label='Prediction')
plt.plot(x[0].numpy(), true, label='True')
plt.legend()
plt.show()
```

### 代码解释
1. **SpectralConv1d**：这是FNO的核心，通过傅里叶变换（FFT）将输入转为频域，只对低频模式应用学到的权重，然后逆变换回空间域。这使得模型高效捕捉全局依赖。
2. **FNO1d**：堆叠谱卷积层和线性层，使用GELU激活。输入是离散化的函数（网格点上的值），输出也是函数。
3. **数据生成**：用正弦函数作为输入，其积分作为目标，模拟函数映射。
4. **训练**：用MSE损失最小化预测和真实积分的差异。运行后，你可以看到预测曲线接近真实积分。

这个例子是最简单的演示，实际应用中可扩展到PDE求解（如Burgers方程）。如果运行代码，需要PyTorch环境；训练后，损失会下降，预测会匹配目标。
