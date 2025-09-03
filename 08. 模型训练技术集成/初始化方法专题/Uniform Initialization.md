# Uniform Initialization（均匀初始化）



## 定📖 义

**Uniform Initialization** 是深度学习中一种参数初始化方法，它将神经网络的权重初始化为在一个均匀分布区间 $[a, b]$ 内随机采样的值。
其核心思想是通过合理选择区间范围，避免网络在训练开始时出现梯度消失或梯度爆炸的问题。



## 📖 数学描述

如果权重 $W$ 来自均匀分布：

$$
W \sim U(-a, a)
$$

那么每个权重满足：

$$
P(W = x) = \frac{1}{2a}, \quad x \in [-a, a]
$$

常见的区间范围 $a$ 定义方式有以下几种（根据初始化策略不同）：

1. **简单均匀分布**：
   手动指定一个常数范围，如：

$$
W \sim U(-0.05, 0.05)
$$

2. **Xavier/Glorot Uniform Initialization**（常用于Sigmoid/Tanh 激活）：

$$
W \sim U\Big(-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}}, \; \sqrt{\frac{6}{n_\text{in}+n_\text{out}}}\Big)
$$

   其中 $n_\text{in}$ 是输入维度， $n_\text{out}$ 是输出维度。

3. **He/Kaiming Uniform Initialization**（常用于ReLU 激活）：

$$
W \sim U\Big(-\sqrt{\frac{6}{n_\text{in}}}, \; \sqrt{\frac{6}{n_\text{in}}}\Big)
$$



## 📖 最简单的代码例子

### PyTorch 例子

```python
import torch
import torch.nn as nn

# 定义一个简单的线性层
linear = nn.Linear(3, 2)

# 使用 Uniform 初始化 [-0.1, 0.1]
nn.init.uniform_(linear.weight, a=-0.1, b=0.1)
nn.init.zeros_(linear.bias)

print("权重:", linear.weight)
print("偏置:", linear.bias)
```

### NumPy 例子

```python
import numpy as np

# 输入维度=3, 输出维度=2
n_in, n_out = 3, 2

# Xavier Uniform 范围
limit = np.sqrt(6 / (n_in + n_out))

# 从均匀分布采样
weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

print("初始化权重:\n", weights)
```


