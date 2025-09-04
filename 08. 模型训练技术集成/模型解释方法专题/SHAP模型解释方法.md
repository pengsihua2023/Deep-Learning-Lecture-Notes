

# SHAP（SHapley Additive exPlanations）模型解释方法

## 1. 什么是 SHAP？

SHAP（SHapley Additive exPlanations）是一种常用的 **模型解释方法**，基于博弈论中的 **Shapley 值** 概念。
它能量化每个输入特征对预测结果的贡献，从而让“黑箱模型”（如深度学习、树模型、集成方法）变得更透明。


## 2. 为什么需要 SHAP？

* **开发者**：帮助发现特征的重要性，调试模型。
* **用户/业务人员**：提升信任度，理解预测结果。
* **合规/安全**：在金融、医疗等场景下，模型必须具备可解释性。

例子：在一个医疗预测模型中，SHAP 可以解释为什么预测“某个病人风险高”，可能是因为血压偏高、年龄较大等。



## 3. SHAP 的核心原理

### 3.1 基于 Shapley 值

在博弈论中，**Shapley 值**用于衡量每个参与者对合作结果的贡献。
类比到机器学习：

* **玩家** → 特征
* **合作收益** → 模型预测值
* **分配方式** → 每个特征的贡献值

### 3.2 公式

模型输出可以分解为：

$$
f(x) = \phi_0 + \sum_{i=1}^n \phi_i
$$

* $\phi_0$：基准值（通常是训练集预测的平均值）。
* $\phi_i$：特征 $i$ 对预测结果的贡献。

### 3.3 特性

SHAP 值满足以下性质：

1. **公平性**：所有特征的贡献之和等于预测值与基准值之差。
2. **一致性**：若某特征的贡献增大，其 SHAP 值不会变小。
3. **可加性**：贡献值是线性可加的。


## 4. SHAP 在深度学习中的应用

* **局部解释**：解释单个样本的预测。
* **全局解释**：总结所有样本的特征重要性。
* **模型调试**：发现是否依赖了错误或偏差特征。
* **业务应用**：医疗、金融等高风险场景的可解释 AI。


## 5. 可视化方法

SHAP 提供了多种直观的可视化：

* **Summary Plot**：全局特征重要性和取值影响。
* **Force Plot**：单个或多个样本的特征贡献。
* **Dependence Plot**：特征取值与贡献之间的关系。


## 6. 代码实现（PyTorch + SHAP）

下面给出一个 **完整 Notebook Demo**，展示如何训练模型并用 SHAP 解释。

```python
# =========================
# 1. 导入依赖
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 2. 定义模型
# =========================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 3. 生成数据
# =========================
np.random.seed(42)
X_train = np.random.randn(200, 4).astype(np.float32)
y_train = (X_train[:, 0] + 0.5*X_train[:, 1] - X_train[:, 2]) > 0
y_train = y_train.astype(np.float32).reshape(-1, 1)

X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train)

# =========================
# 4. 训练模型
# =========================
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

print("Training finished! Final loss:", loss.item())

# =========================
# 5. 用 SHAP 解释模型
# =========================
explainer = shap.DeepExplainer(model, X_train_torch[:50])   # 背景数据
shap_values = explainer.shap_values(X_train_torch[:50])     # 解释前50个样本

# =========================
# 6. Summary Plot（全局特征重要性）
# =========================
shap.summary_plot(
    shap_values[0], 
    X_train[:50], 
    feature_names=["f1", "f2", "f3", "f4"]
)

# =========================
# 7. Force Plot（单个样本）
# =========================
sample_index = 0
shap.force_plot(
    base_value=explainer.expected_value[0].detach().numpy(),
    shap_values=shap_values[0][sample_index].detach().numpy(),
    features=X_train[sample_index],
    feature_names=["f1","f2","f3","f4"],
    matplotlib=True
)

# =========================
# 8. Force Plot（多个样本）
# =========================
shap.force_plot(
    base_value=explainer.expected_value[0].detach().numpy(),
    shap_values=shap_values[0][:5].detach().numpy(),
    features=X_train[:5],
    feature_names=["f1","f2","f3","f4"],
    matplotlib=True
)

# =========================
# 9. Dependence Plot（特征取值与贡献关系）
# =========================
shap.dependence_plot(
    "f1", 
    shap_values[0], 
    X_train[:50], 
    feature_names=["f1","f2","f3","f4"]
)
```

---

## 7. 总结

* **SHAP = Shapley 值 + 模型解释**
* 能解释单个样本（局部）和整体模型（全局）
* 提供多种可视化方式，直观展示特征贡献
* 在深度学习中常用 `DeepExplainer`，也有适配树模型、线性模型的优化版本

