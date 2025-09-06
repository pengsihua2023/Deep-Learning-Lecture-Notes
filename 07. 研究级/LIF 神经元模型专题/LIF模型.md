# **Leaky Integrate-and-Fire (LIF) 神经元模型**

## 1. 定义

**LIF 神经元**是脉冲神经网络（SNN）中最常见、最简化的模型之一。

* 它把神经元抽象成一个 **带电容和电阻的“电路”**。
* 膜电位 $V(t)$ 会随输入电流逐渐积累（integrate）。
* 膜电位同时会随时间衰减回静息电位（leaky）。
* 当 $V(t)$ 超过阈值 $V_{th}$，神经元发放一个脉冲（spike），然后膜电位被重置。



## 2. 数学描述

LIF 的动力学方程为一阶微分方程：

$$
\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{rest}) + R \cdot I(t)
$$

其中：

* $V(t)$：膜电位
* $V_{rest}$：静息电位
* $R$：膜电阻
* $I(t)$：输入电流
* $\tau_m = RC$：膜时间常数（衰减速度）

**发放规则**：

$$
\text{if } V(t) \geq V_{th} \quad \Rightarrow \quad \text{产生脉冲，且 } V(t) \to V_{reset}
$$

---

## 3. 最小代码例子（Python + NumPy）

下面是一个单个 LIF 神经元的最简模拟，用欧拉法数值积分：

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟参数
T = 100.0   # 总时间 (ms)
dt = 1.0    # 步长 (ms)
steps = int(T/dt)

# LIF 参数
tau_m   = 10.0     # 膜时间常数
V_rest  = -65.0    # 静息电位
V_reset = -65.0    # 复位电位
V_th    = -50.0    # 阈值
R       = 1.0      # 膜电阻
I       = 1.5      # 恒定输入电流

# 状态变量
V = np.zeros(steps)
V[0] = V_rest
spikes = []

# 模拟
for t in range(1, steps):
    dV = (-(V[t-1] - V_rest) + R*I) / tau_m * dt
    V[t] = V[t-1] + dV
    if V[t] >= V_th:        # 触发脉冲
        V[t] = V_reset
        spikes.append(t*dt)

# 画图
time = np.arange(0, T, dt)
plt.plot(time, V, label="膜电位")
plt.axhline(V_th, color='r', linestyle='--', label="阈值")
plt.xlabel("时间 (ms)")
plt.ylabel("电位 (mV)")
plt.title("LIF 神经元模拟")
plt.legend()
plt.show()

print("脉冲发生时间 (ms):", spikes)
```


### 运行效果

* 图像中可以看到膜电位逐渐上升到阈值 → 触发脉冲 → 重置，再次积累 → 形成周期性放电。
* 控制输入电流 `I` 可以决定是否产生脉冲，以及脉冲频率。

