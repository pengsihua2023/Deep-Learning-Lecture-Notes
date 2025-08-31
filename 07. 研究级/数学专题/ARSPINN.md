# AR-SPINN 数学描述

AR-SPINN 结合了 **脉冲神经元（Spiking Neurons）** 的动态、**循环结构（RNN）** 的时间依赖性，以及 **自适应阈值机制（Adaptive Threshold Mechanism）**。
它可以用以下方程来描述：


### 1. 膜电位动态

在时间 \$t\$ 时刻，神经元 \$i\$ 的膜电位更新为：

$$
u_i(t) = \alpha u_i(t-1) + \sum_{j} W_{ij} s_j(t) + \sum_{k} R_{ik} s_k(t-1) - v \cdot s_i(t-1)
$$

其中：

* \$\alpha \in (0,1)\$：泄漏衰减系数
* \$W\_{ij}\$：输入权重
* \$R\_{ik}\$：循环连接权重
* \$s\_j(t) \in {0,1}\$：输入脉冲
* \$s\_i(t-1)\$：前一时刻的自身脉冲
* \$v\$：复位值


### 2. 脉冲函数

$$
s_i(t) = H\big(u_i(t) - \theta_i(t)\big)
$$

其中 \$H(\cdot)\$ 为 Heaviside 阶跃函数，\$\theta\_i(t)\$ 为动态阈值。



### 3. 自适应阈值更新

阈值会随着脉冲历史动态变化：

$$
\theta_i(t) = \theta_0 + \beta \sum_{k < t} \gamma^{t-k} s_i(k)
$$

其中：

* \$\theta\_0\$：基准阈值
* \$\beta > 0\$：适应强度
* \$\gamma \in (0,1)\$：遗忘因子

---

# 最简 Python 实现

下面是一个最简的 **AR-SPINN 单神经元实现**，用来说明其核心思想：

```python
import numpy as np

class ARSPINN:
    def __init__(self, n_inputs, alpha=0.9, v=1.0,
                 theta0=1.0, beta=0.1, gamma=0.95):
        self.alpha = alpha      # 衰减系数
        self.v = v              # 脉冲后的复位幅度
        self.theta0 = theta0    # 初始阈值
        self.beta = beta        # 阈值适应强度
        self.gamma = gamma      # 阈值遗忘因子
        self.u = 0.0            # 膜电位
        self.theta = theta0     # 当前阈值
        self.w_in = np.random.randn(n_inputs)  # 输入权重
        self.w_rec = np.random.randn()         # 自循环权重

    def step(self, x_t, s_prev):
        # 膜电位更新：输入 + 循环 + 衰减
        self.u = self.alpha * self.u \
                 + np.dot(self.w_in, x_t) \
                 + self.w_rec * s_prev
        
        # 是否触发脉冲
        s = 1 if self.u >= self.theta else 0
        
        # 脉冲后复位
        if s:
            self.u -= self.v
        
        # 自适应阈值更新
        self.theta = self.theta0 + self.beta * (self.gamma * (self.theta - self.theta0) + s)
        
        return s

# ====== 测试 ======
T = 20
x_seq = np.random.randint(0, 2, (T, 3))  # 随机输入 (20 步，3 维)
neuron = ARSPINN(n_inputs=3)

s_prev = 0
output = []
for t in range(T):
    s_prev = neuron.step(x_seq[t], s_prev)
    output.append(s_prev)

print("输入序列:\n", x_seq)
print("输出脉冲:\n", output)
```



### 该实现展示了 **AR-SPINN** 的三个核心机制：

1. **时间循环性**：通过 `self.w_rec * s_prev` 引入历史依赖；
2. **脉冲神经元动态**：膜电位的积累、衰减与复位；
3. **自适应阈值**：阈值会随着脉冲历史动态变化。



