# ReLoRA (Restarted Low-Rank Adaptation) 微调方法


## 📖 1. 定义

**ReLoRA** 是在 **LoRA (Low-Rank Adaptation)** 基础上的改进型微调方法，核心思想是：

* **LoRA**：在大型预训练模型（LLM）微调时，不更新完整参数 $W \in \mathbb{R}^{d \times k}$，而是插入低秩矩阵分解 $W + BA$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$。这样显著减少参数和显存开销。
* **ReLoRA**：在训练过程中 **定期将 LoRA 的低秩增量合并到原始权重中，然后重置 LoRA 参数**。

  * 这样可以在保持低秩更新效率的同时，避免长期只依赖低秩近似导致的欠拟合；
  * 同时能通过多次 “重启” 累积更多的信息，提升收敛速度和最终效果。

换句话说，ReLoRA 相当于 **周期性地把 LoRA 学到的知识“吸收到”模型权重里，再给 LoRA 一个新的学习空间**。


## 📖 2. 数学公式

设：

* 原始权重矩阵：$W \in \mathbb{R}^{d \times k}$
* LoRA 参数：$A_t \in \mathbb{R}^{r \times k}, B_t \in \mathbb{R}^{d \times r}$
* 有效权重：

$$
W_t^{\text{eff}} = W + B_t A_t
$$

### ReLoRA 更新步骤：

1. **常规 LoRA 更新**（在一个周期内）：

   $$
   (A_t, B_t) \leftarrow (A_{t-1}, B_{t-1}) - \eta \nabla_{A,B} L(W_{t-1}^{\text{eff}})
   $$

2. **周期性合并**（每隔 $T$ 步）：

   $$
   W \leftarrow W + B_t A_t
   $$

   $$
   A_t, B_t \leftarrow \text{init}() \quad (\text{重新随机初始化})
   $$

这样，模型权重 $W$ 会不断吸收 LoRA 的低秩改进，而 LoRA 参数则不断重启，避免训练早期的限制。


## 📖 3. 最简代码例子

下面给出一个极简的 **PyTorch ReLoRA 微调示意代码**（仅演示机制，不是完整库实现）：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ===== 简单的 LoRA 模块 =====
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))  # 原始权重
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)   # LoRA A
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)  # LoRA B
        self.rank = rank

    def forward(self, x):
        return nn.functional.linear(x, self.W + self.B @ self.A)

    def merge_lora(self):
        """把 LoRA 融合进主权重"""
        with torch.no_grad():
            self.W += self.B @ self.A
            nn.init.normal_(self.A, std=0.01)
            nn.init.normal_(self.B, std=0.01)

# ===== 数据和模型 =====
x = torch.randn(100, 10)
y = torch.randn(100, 5)

model = LoRALinear(10, 5, rank=4)
criterion = nn.MSELoss()
optimizer = optim.Adam([model.A, model.B], lr=1e-2)  # 只训练 LoRA 参数

# ===== ReLoRA 训练 =====
steps = 200
merge_every = 50  # 每隔 50 步合并一次

for step in range(steps):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (step + 1) % merge_every == 0:
        print(f"Step {step+1}: Loss = {loss.item():.4f}, merging LoRA...")
        model.merge_lora()

```


### 📖 解释

1. **LoRALinear**：实现了一个带 LoRA 的线性层。
2. **merge\_lora()**：把 $BA$ 融合进主权重 $W$，然后重新初始化 $A, B$。
3. **训练循环**：每隔 `merge_every` 步调用一次 `merge_lora()`，实现 ReLoRA 的周期性“重启”。
4. **效果**：相比单纯 LoRA，ReLoRA 可以获得更稳定的收敛效果。

## ReLoRA vs LoRA 收敛效果对比的示例代码。
我们用一个 **简单的回归任务**，对比两者在相同条件下的损失下降情况。

## ReLoRA vs LoRA 对比实验

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===== LoRA 模块 =====
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        return nn.functional.linear(x, self.W + self.B @ self.A)

    def merge_lora(self):
        """把 LoRA 融合进主权重，并重置 A,B"""
        with torch.no_grad():
            self.W += self.B @ self.A
            nn.init.normal_(self.A, std=0.01)
            nn.init.normal_(self.B, std=0.01)

# ===== 数据 =====
torch.manual_seed(42)
x = torch.randn(200, 10)
true_w = torch.randn(5, 10)
y = x @ true_w.T + torch.randn(200, 5) * 0.1  # 线性任务 + 少量噪声

# ===== 实验配置 =====
steps = 300
merge_every = 50

def train(model, relora=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([model.A, model.B], lr=1e-2)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if relora and (step + 1) % merge_every == 0:
            model.merge_lora()

        losses.append(loss.item())
    return losses

# ===== 训练 LoRA 和 ReLoRA =====
model_lora = LoRALinear(10, 5, rank=4)
losses_lora = train(model_lora, relora=False)

model_relora = LoRALinear(10, 5, rank=4)
losses_relora = train(model_relora, relora=True)

# ===== 绘图 =====
plt.plot(losses_lora, label="LoRA")
plt.plot(losses_relora, label="ReLoRA")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("LoRA vs ReLoRA 收敛对比")
plt.legend()
plt.show()
```



## 📖 代码说明

1. **数据**：构造了一个线性任务 $y = Wx + \epsilon$。
2. **LoRALinear**：和之前一样，实现 LoRA 权重。
3. **LoRA 训练**：只更新低秩矩阵 $A, B$。
4. **ReLoRA 训练**：在训练中每隔 `merge_every=50` 步执行一次 `merge_lora()`。
5. **结果**：绘制 LoRA 和 ReLoRA 的收敛曲线。

在实际运行中你会看到：

* **LoRA** 曲线下降，但有时会收敛得比较慢或停滞；
* **ReLoRA** 曲线下降更稳定，能达到更低的 loss。



