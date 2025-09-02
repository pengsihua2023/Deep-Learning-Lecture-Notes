# He Initialization（也叫 Kaiming Initialization）



## 定义

**He Initialization** 是一种专门为 **ReLU 及其变种激活函数**（如 ReLU、LeakyReLU、ELU 等）设计的参数初始化方法。
它的目标是：

* 保持前向传播时不同层的输出方差稳定；
* 减少梯度消失或梯度爆炸问题。

He 初始化方法由何凯明（Kaiming He）等人在 2015 年提出（论文 *Delving Deep into Rectifiers*）。



## 数学描述

设某一层的输入维度为 $n_\text{in}$（即输入神经元数），则 He 初始化分为 **均匀分布版** 和 **正态分布版**：

1. **He Normal (正态分布版)**

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_\text{in}}\right)
$$

2. **He Uniform (均匀分布版)**

$$
W \sim U\left(-\sqrt{\frac{6}{n_\text{in}}}, \; \sqrt{\frac{6}{n_\text{in}}}\right)
$$

其中：

* $\mathcal{N}(0, \sigma^2)$ 表示均值为 0、方差为 $\sigma^2$ 的正态分布；
* $U(-a, a)$ 表示区间 $[-a, a]$ 上的均匀分布。

> 核心思想：
> 因为 ReLU 会截断一半输入为 0，所以需要 **放大权重方差**，使用 $\frac{2}{n_\text{in}}$ 而不是 Xavier 初始化中的 $\frac{1}{n_\text{in}}$。



## 最简单的代码例子

### PyTorch 例子

```python
import torch
import torch.nn as nn

# 定义一个线性层 (输入=3, 输出=2)
linear = nn.Linear(3, 2)

# He Uniform 初始化
nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
nn.init.zeros_(linear.bias)

print("He Uniform 权重:\n", linear.weight)
print("偏置:\n", linear.bias)

# He Normal 初始化（如果想用正态分布）
linear2 = nn.Linear(3, 2)
nn.init.kaiming_normal_(linear2.weight, nonlinearity='relu')
print("He Normal 权重:\n", linear2.weight)
```

### NumPy 例子

```python
import numpy as np

n_in, n_out = 3, 2

# He Uniform
limit = np.sqrt(6 / n_in)
weights_uniform = np.random.uniform(-limit, limit, size=(n_out, n_in))

# He Normal
std = np.sqrt(2 / n_in)
weights_normal = np.random.normal(0, std, size=(n_out, n_in))

print("He Uniform 初始化:\n", weights_uniform)
print("He Normal 初始化:\n", weights_normal)
```


