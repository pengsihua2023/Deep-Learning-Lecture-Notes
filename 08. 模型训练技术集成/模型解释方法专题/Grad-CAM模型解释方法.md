# Grad-CAM 模型解释方法 (Gradient-weighted Class Activation Mapping)

## 1. 定义

**Grad-CAM** 是一种 **可视化神经网络预测依据** 的方法，常用于 CNN 图像分类模型的解释。

* 通过计算目标类别的 **梯度对卷积特征图的影响**，得到各通道的重要性权重；
* 再将这些权重加权特征图，得到一张 **类激活热力图**；
* 该热力图显示了模型在输入图像中最关注的区域。

它相比 **Saliency Maps** 更平滑，更易于人类理解。


## 2. 数学描述

设：

* 输入图像 $x$，
* CNN 最后一个卷积层的特征图为 $A^k \in \mathbb{R}^{H \times W}$ ，其中 $k$ 为通道索引，
* 对类别 $c$ 的预测分数（logit）为 $y^c$ 。

**Grad-CAM 的步骤**：

1. **计算权重**：对类别 $c$ 的 logit 对特征图 $A^k$ 求梯度，并在空间维度上取平均：

$$
\alpha_k^c = \frac{1}{H \times W} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

2. **生成热力图**：

$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$

3. **上采样**到原图大小并叠加在输入图像上。



## 3. 最简代码例子

用 **PyTorch** 在 ResNet18 上实现 Grad-CAM：

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===== 预训练模型 =====
model = models.resnet18(pretrained=True)
model.eval()

# ===== 输入图像处理 =====
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
img = Image.open("cat.jpg")  # 输入一张图片
x = preprocess(img).unsqueeze(0)

# ===== Hook: 捕获最后一层卷积的特征图和梯度 =====
features = []
grads = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])

layer = model.layer4[1].conv2  # ResNet18最后一个卷积层
layer.register_forward_hook(forward_hook)
layer.register_backward_hook(backward_hook)

# ===== 前向预测 =====
output = model(x)
pred_class = output.argmax(dim=1).item()

# ===== 反向传播，获取梯度 =====
model.zero_grad()
score = output[0, pred_class]
score.backward()

# ===== 计算 Grad-CAM =====
feature_map = features[0].squeeze(0)      # [C,H,W]
grad = grads[0].squeeze(0)                # [C,H,W]
weights = grad.mean(dim=(1,2))            # [C]

cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)
for i, w in enumerate(weights):
    cam += w * feature_map[i]

cam = F.relu(cam)
cam = cam / cam.max()  # 归一化到 [0,1]

# ===== 可视化 =====
plt.subplot(1,2,1)
plt.title("原始图像")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Grad-CAM")
plt.imshow(img)
plt.imshow(cam.detach().numpy(), cmap='jet', alpha=0.5)  # 热力图叠加
plt.show()
```

## 结果说明

* **左图**：原始输入图像（例如猫）。
* **右图**：Grad-CAM 热力图，红色区域表示模型最关注的部分（比如猫的脸）。



## 总结

* **Grad-CAM 定义**：基于梯度的类激活可视化方法，解释 CNN 预测。
* **公式**：$\alpha_k^c = \frac{1}{H \times W}\sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}},\; L^c = ReLU(\sum_k \alpha_k^c A^k)$。
* **代码**：只需 Hook 特征图和梯度，再加权生成热力图。


