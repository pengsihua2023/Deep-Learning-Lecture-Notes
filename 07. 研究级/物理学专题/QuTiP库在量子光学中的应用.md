## QuTiP库在量子光学中的应用
以下是一个使用QuTiP库的代码例子，基于真实数据。该例子模拟Jaynes-Cummings模型（量子光学中的标准模型，描述二能级原子与单模腔场的相互作用），使用真空Rabi振荡的动力学演化。参数来源于真实量子光学实验，如超导电路QED系统中的典型值（例如，腔频率wc ≈ 5 GHz，耦合强度g ≈ 100 MHz，衰减率κ ≈ 1 MHz，原子衰减γ ≈ 0.1 MHz，这些值基于文献中报告的实验数据，如电路量子电动力学实验）。 代码计算开放系统下的时间演化，并绘制腔和原子的占据概率。

### 完整代码例子
```python
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# 真实实验参数（基于超导QED实验，单位：2π * Hz）
N = 15  # 腔Fock态截断维度
wc = 2 * np.pi * 5e9  # 腔频率 ≈ 5 GHz
wa = 2 * np.pi * 5e9  # 原子频率 ≈ 5 GHz（共振情况）
g = 2 * np.pi * 100e6  # 耦合强度 ≈ 100 MHz
kappa = 2 * np.pi * 1e6  # 腔衰减率 ≈ 1 MHz
gamma = 2 * np.pi * 0.1e6  # 原子衰减率 ≈ 0.1 MHz
n_th = 0.01  # 热浴平均激发数（低温下接近0）

# 操作符定义
a = qt.tensor(qt.destroy(N), qt.qeye(2))  # 腔湮灭算符
sm = qt.tensor(qt.qeye(N), qt.sigmam())  # 原子降低算符
sz = qt.tensor(qt.qeye(N), qt.sigmaz())  # 原子Pauli Z算符

# Jaynes-Cummings哈密顿量（使用旋转波近似）
H = wc * a.dag() * a + (wa / 2) * sz + g * (a.dag() * sm + a * sm.dag())

# 初始态：原子激发，腔真空
psi0 = qt.tensor(qt.basis(N, 0), qt.basis(2, 0))  # 腔|0>，原子|1>

# 时间列表（单位：s，覆盖几个Rabi周期）
tlist = np.linspace(0, 1e-6, 500)  # 0到1 μs

# 坍缩算符（开放系统）
c_ops = [
    np.sqrt(kappa * (1 + n_th)) * a,  # 腔衰减
    np.sqrt(kappa * n_th) * a.dag(),   # 腔热激发
    np.sqrt(gamma) * sm                # 原子弛豫
]

# 求解主方程
result = qt.mesolve(H, psi0, tlist, c_ops, [a.dag() * a, sz / 2 + 0.5])  # 期望值：腔光子数，原子激发概率

# 绘制结果
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist * 1e9, result.expect[0], label='Cavity photon number')
ax.plot(tlist * 1e9, result.expect[1], label='Atom excitation probability')
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Occupation probability')
ax.set_title('Vacuum Rabi Oscillations in Jaynes-Cummings Model')
ax.legend()
plt.show()
```

### 代码说明
- **数据来源**：参数基于真实实验，如超导电路中的腔量子电动力学（CQED）系统，这些值常见于文献报告的实验结果（如Rabi频率2g ≈ 200 MHz，衰减率匹配实验测量）。
- **运行要求**：需安装QuTiP（pip install qutip）。运行后会生成振荡图，显示光子和原子激发态的交换（Rabi振荡），受衰减影响逐渐衰减。
- **应用**：此模拟可用于分析量子光学实验中的相干演化，与真实腔QED系统一致。

