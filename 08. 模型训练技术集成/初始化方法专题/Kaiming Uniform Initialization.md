# Kaiming Uniform Initialization（何凯明均匀初始化）



## 定义

**Kaiming Uniform Initialization** 是 He Initialization 的 **均匀分布版本**，用于 ReLU 及其变体激活函数（如 ReLU、LeakyReLU）。
它的目标是让 **前向传播时每一层的输出方差保持稳定**，避免梯度消失或梯度爆炸。

Kaiming 初始化由何凯明（Kaiming He）等人在 2015 年论文 *Delving Deep into Rectifiers* 提出。

---

## 数学描述

假设某一层的输入维度为 $n_\text{in}$（即 $fan\_in$ ，输入神经元个数），则权重 $W$ 来自区间：

$$
W \sim U\left(-\text{bound}, \; \text{bound}\right)
$$

其中：

$$
\text{bound} = \sqrt{\frac{6}{n_\text{in} \cdot (1 + a^2)}}
$$

* $a$ 是 ReLU 的 **负半轴斜率**（对于标准 ReLU， $a = 0$ ；对于 Leaky ReLU， $a$ 是泄露系数）。
* 当 $a = 0$（标准 ReLU）时，公式化简为：

$$
W \sim U\left(-\sqrt{\frac{6}{n_\text{in}}}, \; \sqrt{\frac{6}{n_\text{in}}}\right)
$$

> 这个范围比 Xavier Uniform 更大，因为 ReLU 会丢掉一半输入（负数部分变 0），所以要加大权重的方差。



## 最简单的代码例子

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 定义一个线性层 (输入=3, 输出=2)
linear = nn.Linear(3, 2)

# 使用 Kaiming Uniform 初始化 (对应 He Uniform)
nn.init.kaiming_uniform_(linear.weight, a=0, nonlinearity='relu')
nn.init.zeros_(linear.bias)

print("Kaiming Uniform 权重:\n", linear.weight)
print("偏置:\n", linear.bias)
```

### NumPy 实现

```python
import numpy as np

n_in, n_out = 3, 2
a = 0  # ReLU 的负半轴斜率

bound = np.sqrt(6 / (n_in * (1 + a**2)))
weights = np.random.uniform(-bound, bound, size=(n_out, n_in))

print("Kaiming Uniform 初始化权重:\n", weights)
```


