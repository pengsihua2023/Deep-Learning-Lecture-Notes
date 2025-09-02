# SparseAdam 优化器

## 1. 定义

**SparseAdam** 是 **Adam** 的一个变体，专门用于处理 **稀疏梯度（sparse gradients）** 的场景，典型应用就是 **embedding 层**（如 NLP 的词向量训练）。

普通 Adam 会对所有参数维护一阶、二阶矩的动量估计（`m, v`），即使某些参数本轮没有更新也会计算，从而造成额外开销。而 SparseAdam：

* **只更新非零梯度对应的参数和动量项**，避免了无意义的更新；
* 对未更新的参数，其动量保持不变（不会衰减为 0）。

因此在大规模稀疏梯度场景（如词表上百万规模），SparseAdam 能显著提高效率。


## 2. 数学公式

设：

* 参数： $\theta_t$
* 梯度： $g_t$（稀疏，即大部分分量为 0）
* 一阶矩（动量）： $m_t$
* 二阶矩： $v_t$
* 学习率： $\alpha$
* 衰减系数： $\beta_1, \beta_2 \in [0,1)$
* 数值稳定项： $\epsilon$

### 更新步骤：

1. 梯度计算（稀疏）：

$$
g_t = \nabla_\theta f_t(\theta_{t-1}) \quad (\text{仅非零部分})
$$

2. 一阶与二阶矩更新（仅非零索引）：

$$
m_t[i] = \beta_1 m_{t-1}[i] + (1-\beta_1) g_t[i]
$$

$$
v_t[i] = \beta_2 v_{t-1}[i] + (1-\beta_2) g_t[i]^2
$$

3. 偏差修正（bias correction）：

$$
\hat{m}_t[i] = \frac{m_t[i]}{1-\beta_1^t}, \quad 
\hat{v}_t[i] = \frac{v_t[i]}{1-\beta_2^t}
$$

4. 参数更新：

$$
\theta_t[i] = \theta_{t-1}[i] - \alpha \cdot \frac{\hat{m}_t[i]}{\sqrt{\hat{v}_t[i]} + \epsilon}
$$

这里的 $i$ 表示只有 **非零梯度对应的参数索引** 被更新。


## 3. 最简代码例子

用 **PyTorch 的 SparseAdam** 在一个嵌入层上做演示：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模拟一个嵌入层 (词表大小=1000, 维度=50)
embedding = nn.Embedding(1000, 50, sparse=True)

# 随机生成一些词ID (batch_size=4, seq_len=5)
input_ids = torch.randint(0, 1000, (4, 5))

# 前向传播 (得到 embedding 向量)
embeddings = embedding(input_ids)

# 定义一个简单的目标 (最小化 embedding 向量的 L2 范数)
loss = embeddings.pow(2).sum()

# 优化器：SparseAdam
optimizer = optim.SparseAdam(embedding.parameters(), lr=0.01)

# 反向传播 & 更新
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("更新后的嵌入向量(部分):")
print(embedding.weight[input_ids[0]])
```



### 解释

1. **Embedding 层** 设置了 `sparse=True`，梯度就是稀疏的。
2. **SparseAdam** 会只更新本次 batch 中出现的 token 对应的 embedding。
3. 其他没出现的词向量保持不变，效率更高。


