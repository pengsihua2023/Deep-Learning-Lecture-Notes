## 基于QuTiP库谐振子的动力学演化研究
以下是一个使用QuTiP库的代码例子，基于量子谐振子的真实数据。该例子模拟一个开放量子谐振子的动力学演化（受衰减影响的相干态），参数来源于量子光学实验中的典型光学腔模式（如微波超导腔实验，腔频率ω ≈ 2π * 5 GHz，衰减率κ ≈ 2π * 1 MHz，热浴平均光子数n_th ≈ 0.01，这些值基于文献报告的实验数据，如电路量子电动力学中的谐振腔测量）。代码计算系统的时间演化，并绘制光子数期望值。

### 完整代码例子
```python
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# 真实实验参数（基于量子光学腔实验，单位：2π * Hz）
N = 20  # Fock态截断维度
omega = 2 * np.pi * 5e9  # 谐振频率 ≈ 5 GHz
kappa = 2 * np.pi * 1e6  # 衰减率 ≈ 1 MHz
n_th = 0.01  # 热浴平均光子数（低温下接近0）

# 操作符定义
a = qt.destroy(N)  # 湮灭算符

# 哈密顿量（量子谐振子）
H = omega * a.dag() * a

# 初始态：相干态（alpha=2，代表初始光子数|alpha|^2=4）
alpha = 2.0
psi0 = qt.coherent(N, alpha)

# 时间列表（单位：s，覆盖几个相干周期）
tlist = np.linspace(0, 1e-7, 500)  # 0到100 ns

# 坍缩算符（开放系统，考虑衰减和热浴）
c_ops = [
    np.sqrt(kappa * (1 + n_th)) * a,  # 衰减
    np.sqrt(kappa * n_th) * a.dag()    # 热激发
]

# 求解主方程
result = qt.mesolve(H, psi0, tlist, c_ops, [a.dag() * a])  # 期望值：光子数

# 绘制结果
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist * 1e9, result.expect[0], label='Photon number')
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Expectation value')
ax.set_title('Dynamics of Open Quantum Harmonic Oscillator (Coherent State Decay)')
ax.legend()
plt.show()
```

### 代码说明
- **数据来源**：参数基于真实量子光学实验，如超导腔中的谐振模式，这些值常见于文献报告的测量结果（例如，腔频率在GHz范围，衰减率在MHz范围，导致相干时间在μs级）。
- **运行要求**：需安装QuTiP（pip install qutip）。运行后会生成图表，显示光子数随时间衰减（由于κ引起的耗散）。
- **应用**：此模拟可用于分析量子光学中的腔模式衰减，与真实实验（如噪声谱测量）一致。
