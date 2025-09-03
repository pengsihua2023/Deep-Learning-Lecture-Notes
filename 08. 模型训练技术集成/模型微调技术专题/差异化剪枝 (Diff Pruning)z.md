
# 差异化剪枝 (Diff Pruning)

## 📖 1. 定义

**差异化剪枝（Diff Pruning）** 是一种 **参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）** 方法，最早用于大规模预训练语言模型的高效适配（Guo et al., 2021）。

核心思想：

* 不直接修改预训练模型的参数 $\theta$，而是为每个参数引入一个 **差分参数（diff parameter）** $\Delta \theta$。
* 在下游任务中只学习这些差分参数，而不是整个模型。
* 通过 **稀疏化约束（如 L1 正则或门控机制）**，仅在必要的参数位置引入差分更新，达到 **高效 + 可解释** 的效果。

👉 简单来说：
模型参数更新方式从：

$$
\theta' = \theta + \Delta \theta
$$

转变为：

* **多数位置**： $\Delta \theta = 0$ （即冻结，不更新）。
* **少数位置**： $\Delta \theta \neq 0$ （只更新必要的参数）。


## 📖 2. 数学描述

### 2.1 参数表示

设：

* 预训练参数：$\theta \in \mathbb{R}^d$
* 差分参数：$\Delta \theta \in \mathbb{R}^d$
* 下游模型参数：

$$
\theta' = \theta + \Delta \theta
$$

### 2.2 损失函数

训练目标是最小化下游任务的损失，同时让差分参数稀疏：

$$
\mathcal{L}(\Delta \theta) = \mathcal{L}_{task}(f(x; \theta + \Delta \theta)) + \lambda \|\Delta \theta\|_1
$$

* $\mathcal{L}_{task}$：下游任务损失（如分类交叉熵）。
* $\|\Delta \theta\|_1$：L1 正则化，鼓励稀疏性。
* $\lambda$：正则系数。

### 2.3 剪枝机制

* 在训练过程中，很多 $\Delta \theta$ 会收敛到接近 0。
* 最终可以将这些位置剪枝，只保留少量非零参数。

## 📖 3. 简单代码示例（PyTorch）

下面用 PyTorch 实现一个 **线性层的差异化剪枝示例**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性分类模型
class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        # 冻结预训练参数
        for p in self.fc.parameters():
            p.requires_grad = False
        # 差分参数（可训练）
        self.delta = nn.Parameter(torch.zeros_like(self.fc.weight))

    def forward(self, x):
        # 原始参数 + 差分参数
        return (self.fc.weight + self.delta) @ x.T

# 模拟数据
X = torch.randn(10, 5)   # batch=10, input_dim=5
y = torch.randint(0, 2, (10,))  # 二分类标签

# 初始化模型
model = BaseModel(input_dim=5, output_dim=2)

# 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([model.delta], lr=0.01)

# 训练
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.T, y)
    # L1 正则化约束差分参数
    loss = loss + 0.01 * torch.norm(model.delta, p=1)
    loss.backward()
    optimizer.step()

print("Trained Δθ (sparse updates):")
print(model.delta)
```


## 📖 4. 总结

* **定义**：Diff Pruning 通过学习 **稀疏差分参数** 来高效微调预训练模型。
* **数学公式**：

$$
\theta' = \theta + \Delta \theta, \quad 
\mathcal{L} = \mathcal{L}_{task} + \lambda \|\Delta \theta\|_1
$$
* **特点**：

  * 节省显存和计算量（只更新少量参数）。
  * 保留预训练模型的泛化能力。
  * 剪枝后得到稀疏可解释的差分更新。
* **代码**：通过冻结原始参数，仅训练 $\Delta \theta$ ，并加上 L1 正则实现稀疏化。


