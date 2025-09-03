# Saliency Maps（显著性图）模型解释方法

## 1. 定义

**Saliency Map** 是一种 **基于梯度的模型解释方法**，最早用于图像分类。
它通过计算 **输入像素对输出预测的梯度大小**，来衡量哪些输入特征对模型预测最重要。

* 如果一个像素的梯度较大，说明对预测结果的影响更大；
* 显著性图通常以热力图的形式可视化，突出最关键的区域。

这种方法适用于 **神经网络图像模型**，也可扩展到文本和表格任务。



## 2. 数学描述

设：

* 输入图像向量 $x \in \mathbb{R}^d$
* 目标类别的预测分数（logit）为 $S_c(x)$

显著性图定义为：

$$
M(x) = \left| \frac{\partial S_c(x)}{\partial x} \right|
$$

其中：

* $\frac{\partial S_c(x)}{\partial x}$：对输入图像每个像素求梯度；
* 取绝对值或平方，表示该像素的重要性。

最终，显著性图 $M(x)$ 可以映射成一张热力图，直观展示模型关注的区域。



## 3. 最简代码例子

用 **PyTorch** 在 MNIST 上生成一张显著性图：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== 简单 CNN 模型 =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26*26*16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# ===== 加载数据 & 模型 =====
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
x, y = dataset[0]   # 取第一张图
x = x.unsqueeze(0)  # [1,1,28,28]

model = SimpleCNN()
model.eval()

# ===== 前向传播 =====
x.requires_grad_()  # 允许对输入求梯度
output = model(x)
pred_class = output.argmax(dim=1).item()

# ===== 计算梯度 =====
score = output[0, pred_class]  # 目标类别分数
score.backward()  # 反向传播
saliency = x.grad.data.abs().squeeze().numpy()  # 显著性图

# ===== 可视化 =====
plt.subplot(1,2,1)
plt.title(f"原始图像 (label={y})")
plt.imshow(x.detach().squeeze(), cmap="gray")

plt.subplot(1,2,2)
plt.title("Saliency Map")
plt.imshow(saliency, cmap="hot")
plt.show()
```


## 结果说明

* **左图**：原始 MNIST 数字（如 “7”）。
* **右图**：显著性图，显示哪些像素对模型预测“7”贡献最大（一般是数字的边缘）。



## 总结

* **定义**：Saliency Maps 用输入梯度解释模型预测。
* **公式**： $M(x) = \left| \frac{\partial S_c(x)}{\partial x} \right|$。
* **代码**：只需几行 PyTorch 梯度操作即可生成显著性图。

---

## Saliency Maps / LIME / SHAP 对比

| 特点        | **Saliency Maps**             | **LIME**                                   | **SHAP**                              |                                                             |                                                       |   |      |   |                               |
| --------- | ----------------------------- | ------------------------------------------ | ------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- | - | ---- | - | ----------------------------- |
| **原理**    | 基于梯度：计算预测分数对输入的偏导数            | 在输入邻域生成扰动样本，用可解释模型（线性/树）近似黑箱模型             | 基于博弈论 Shapley 值，计算特征的边际贡献             |                                                             |                                                       |   |      |   |                               |
| **公式**    | (M(x) = $\left                 | \frac{\partial S\_c(x)}{\partial x} \right | )                                     | $\arg\min_g \sum_i \pi_x(z_i)(f(z_i)-g(z_i))^2 + \Omega(g)$ | (\phi\_i = \sum\_{S \subseteq F \setminus {i}} \frac{ | S | !(M- | S | -1)!}{M!}\[f(S\cup{i})-f(S)]) |
| **解释范围**  | **局部解释**，针对某个输入样本，显示哪些像素/特征重要 | **局部解释**，针对某个输入样本，说明哪些特征推动预测               | **局部 + 全局解释**，每个特征的 Shapley 值可聚合成全局解释 |                                                             |                                                       |   |      |   |                               |
| **模型依赖性** | 依赖梯度 → 只能用于 **可微分模型**（如神经网络）  | **完全模型无关**，只需预测接口                          | **模型无关**，但有专门高效版本（TreeSHAP、DeepSHAP）  |                                                             |                                                       |   |      |   |                               |
| **输出形式**  | 热力图（像素级），突出最重要区域              | 特征权重列表（正/负影响）                              | 特征贡献值（可加和，公平分配预测）                     |                                                             |                                                       |   |      |   |                               |
| **稳定性**   | 不稳定（受噪声和梯度消失影响）               | 不稳定（不同采样可能导致不同解释）                          | 稳定（Shapley 值唯一，满足公平性公理）               |                                                             |                                                       |   |      |   |                               |
| **计算成本**  | 较低（一次反向传播即可）                  | 中等（需要采样和拟合线性模型）                            | 较高（原始算法指数级，但有近似/优化版本）                 |                                                             |                                                       |   |      |   |                               |
| **适用场景**  | 计算机视觉（图像分类、医学影像）              | NLP、表格、图像等通用场景                             | 高风险任务（金融、医疗），需要严格可信解释                 |                                                             |                                                       |   |      |   |                               |


## 总结

* **Saliency Maps**：梯度法，适合图像模型，速度快但不稳定。
* **LIME**：基于采样 + 可解释模型，直观灵活，但可能不稳定。
* **SHAP**：理论最强，稳定可信，但计算复杂度较高。



