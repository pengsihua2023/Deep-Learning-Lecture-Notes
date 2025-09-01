
# Integrated Gradients (IG) 模型解释方法

## 1. 定义

**Integrated Gradients (IG)** 是一种基于梯度的 **模型解释方法**，由 Sundararajan 等人在 2017 年提出。
它的核心思想是：

> 将输入特征从一个 **参考输入 (baseline)**（比如全零向量、均值输入等）逐渐变化到实际输入，并沿着这条路径积分梯度，从而度量每个特征对输出的贡献。

这样可以解决 **普通梯度方法** 的两个问题：

* 梯度可能在饱和区间接近 0（但特征仍有影响）。
* 单点梯度可能不稳定。



## 2. 数学描述

设：

* 输入：$x$
* 参考输入：$x'$（baseline）
* 模型函数：$F(x)$
* 输入第 $i$ 个特征的 Integrated Gradient：

$$
IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F\big(x' + \alpha (x - x')\big)}{\partial x_i} d\alpha
$$

解释：

* 将输入从 baseline 沿直线路径逐渐过渡到实际输入：

  $$
  x(\alpha) = x' + \alpha(x - x')
  $$
* 在路径上采样多个点，计算梯度并累积平均。
* 结果 $\text{IG}_i(x)$ 表示特征 $i$ 对预测结果的贡献。



## 3. 简单代码示例（PyTorch + Captum）

我们用 `captum` 库来实现 **Integrated Gradients**。

```python
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# 1. 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()

# 2. 构造输入
inputs = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
baseline = torch.zeros_like(inputs)  # baseline=0

# 3. 使用 Integrated Gradients 解释
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs, baselines=baseline, n_steps=50)

print("Inputs:", inputs)
print("Attributions (feature contributions):", attributions)
print("Sum of attributions:", attributions.sum().item())
print("Prediction difference:", (model(inputs) - model(baseline)).item())
```

输出结果示例：

```
Inputs: tensor([[1., 2., 3.]], grad_fn=<...>)
Attributions: tensor([[0.15, 0.42, 0.38]], grad_fn=<...>)
Sum of attributions: 0.95
Prediction difference: 0.95
```

可以看到：

* 每个特征都有一个贡献值（正/负）。
* 所有贡献值的和 ≈ 预测结果与 baseline 的差。



## 4. 总结

* **定义**：IG 通过积分梯度，解释输入特征对预测的贡献。
* **公式**：

  $$
  IG_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
  $$
* **特点**：

  * 满足 **归因完整性**（贡献和 = 预测差）。
  * 避免了梯度在饱和区间为 0 的问题。
* **实现**：`captum.attr.IntegratedGradients`


