# LoKr (Low-rank Kronecker Product Adaptation)** 微调方法

## 1. 定义

**LoKr** 是一种参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）方法，和 **LoRA**、**LoHA** 类似，核心思想是在预训练大模型参数的基础上，**冻结原始权重**，只学习一个低秩的更新矩阵 $\Delta W$。

不同之处在于：

* **LoRA**： $\Delta W = BA$ ，低秩近似。
* **LoHA**： $\Delta W = (BA) \odot (DC)$ ，用 Hadamard 积增强表达能力。
* **LoKr**： $\Delta W$ 使用 **Kronecker 积 (⊗)** 表示，即：

  $$
  \Delta W = A \otimes B
  $$

  其中 $A, B$ 是较小的矩阵。

这样，LoKr 能用更少的参数表示一个大矩阵更新，适合在更大规模模型上节省显存和计算。



## 2. 数学公式

设原始权重矩阵 $W \in \mathbb{R}^{d \times k}$，冻结不动。LoKr 的低秩更新为：

1. **有效权重**：

$$
W^{\text{eff}} = W + \Delta W
$$

2. **Kronecker 分解形式**：

$$
\Delta W = A \otimes B
$$

其中：

* $A \in \mathbb{R}^{m \times n}$，$B \in \mathbb{R}^{p \times q}$
* Kronecker 积结果 $\Delta W \in \mathbb{R}^{(m p) \times (n q)}$
* 通过选择合适的 $m,n,p,q$，可以近似原始权重维度 $(d \times k)$。

训练时，只更新 $A, B$，其参数量远小于完整的 $d \times k$。



## 3. 最简代码例子

用 **PyTorch** 实现一个极简 LoKr 线性层：

```python
import torch
import torch.nn as nn

class LoKrLinear(nn.Module):
    def __init__(self, in_features, out_features, m=2, n=2):
        super().__init__()
        # 冻结的原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoKr 参数 (两个小矩阵)
        self.A = nn.Parameter(torch.randn(m, n) * 0.01)
        # Kronecker 另一部分矩阵形状推导
        p, q = out_features // m, in_features // n
        self.B = nn.Parameter(torch.randn(p, q) * 0.01)

    def forward(self, x):
        # Kronecker 积生成 ΔW
        delta_W = torch.kron(self.A, self.B)  # Kronecker product
        W_eff = self.weight + delta_W
        return x @ W_eff.T

# ===== 测试 =====
x = torch.randn(2, 8)   # 输入 [batch, in_features]
layer = LoKrLinear(8, 4, m=2, n=2)  # out=4, in=8
out = layer(x)
print("输出形状:", out.shape)
```

输出：

```
输出形状: torch.Size([2, 4])
```

说明 LoKr 层正常运行。



## 总结

* **LoKr** 用 Kronecker 积 $(A \otimes B)$ 来构造低秩更新。
* 这样参数量从 $O(dk)$ 降到 $O(mn + pq)$，但仍能近似表达大矩阵更新。
* 适合在大模型（如 LLM）中做参数高效微调。


## 🔹 LoRA / LoHA / LoKr 微调方法对比

| 方法       | 更新公式                           | 额外参数规模                            | 表达能力                    | 特点                         |
| -------- | ------------------------------ | --------------------------------- | ----------------------- | -------------------------- |
| **LoRA** | $\Delta W = B A$               | $O(d r + r k)$                    | 中等（低秩线性近似）              | 最经典的 PEFT 方法，参数量小，效果好，简单高效 |
| **LoHA** | $\Delta W = (B A) \odot (D C)$ | $O(2 (d r + r k))$                | 较强（Hadamard 逐元素乘增强表示）   | 在保持低秩的同时增强非线性建模能力，适合复杂任务   |
| **LoKr** | $\Delta W = A \otimes B$       | $O(m n + p q)$ （远小于 $d \times k$） | 较强（Kronecker 积能表达大矩阵结构） | 参数量极小但能表示大矩阵，特别适合大模型的显存优化  |


## 总结

* **LoRA**：最基础的低秩近似，简单实用，广泛应用于 LLM 微调。
* **LoHA**：在 LoRA 基础上增加 Hadamard 乘积，表示能力更强，适合更复杂的任务。
* **LoKr**：利用 Kronecker 积，用很少的参数近似大矩阵更新，适合在超大模型中进一步节省显存。



