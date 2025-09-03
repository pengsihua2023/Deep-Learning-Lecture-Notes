## 动态调整学习率
### 📖 什么是动态调整学习率（Learning Rate Scheduling）？

动态调整学习率（Learning Rate Scheduling）是深度学习中一种优化技术，通过在训练过程中根据一定策略动态改变学习率，以加速收敛、避免震荡并提高模型性能。学习率决定了参数更新的步长，合适的调度策略可以在训练初期快速下降损失，后期精细调整以接近最优解。

### 📖 核心原理
- **初始学习率**：通常较高（如0.01或0.001），以快速优化。
- **调整策略**：根据训练轮数（epoch）、损失或性能指标调整学习率。
- **常见调度方法**：
  - **时间衰减（Step Decay）**：每隔固定轮数降低学习率（如每5个epoch乘以0.1）。
  - **指数衰减（Exponential Decay）**：学习率按指数逐渐减小。
  - **余弦退火（Cosine Annealing）**：学习率按余弦函数平滑变化。
  - **自适应调整**：如基于验证损失的ReduceLROnPlateau。

### 📖 优势
- **加速收敛**：初期高学习率快速下降，末期低学习率精细调整。
- **防止震荡**：避免学习率过高导致损失函数在最优解附近震荡。
- **提高性能**：动态调整可帮助模型找到更好的局部最优解。

### 📖 局限性
- **超参数选择**：调度策略（如步长、衰减率）需调优。
- **任务依赖**：不同任务可能需要不同调度方法。

---

### 📖 Python代码示例

以下是一个最简单的PyTorch示例，展示如何在MNIST手写数字分类任务中使用**StepLR**（时间衰减）调度器动态调整学习率。代码保持极简，结合Adam优化器。

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# 步骤1: 定义简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # 输入：28x28像素，输出：10类
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = self.fc(x)
        return x

# 步骤2: 加载MNIST数据集
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 步骤3: 初始化模型、损失函数、优化器和调度器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # 每2个epoch学习率乘以0.1

# 步骤4: 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 步骤5: 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 步骤6: 训练循环
epochs = 5
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
    scheduler.step()  # 更新学习率
```

---

### 📖 代码说明

1. **模型定义**：
   - `SimpleNet` 是一个极简的全连接神经网络，输入为MNIST的28x28像素图像，输出为10类分类。

2. **数据集**：
   - 使用`torchvision`加载MNIST数据集，批量大小为64，数据预处理仅包括转换为张量。

3. **学习率调度器**：
   - 使用`StepLR`调度器：`step_size=2`（每2个epoch调整一次），`gamma=0.1`（学习率乘以0.1）。
   - 初始学习率`lr=0.001`，第3个epoch降为0.0001，第5个epoch降为0.00001。

4. **训练与测试**：
   - 训练时打印损失和当前学习率（`optimizer.param_groups[0]["lr"]`）。
   - 每个epoch末调用`scheduler.step()`更新学习率。
   - 测试时计算分类准确率。

5. **输出示例**：
   ```
   Epoch 1, Loss: 0.4321, LR: 0.001000
   Test Accuracy: 92.50%
   Epoch 2, Loss: 0.2876, LR: 0.001000
   Test Accuracy: 93.80%
   Epoch 3, Loss: 0.2345, LR: 0.000100
   Test Accuracy: 94.20%
   Epoch 4, Loss: 0.2234, LR: 0.000100
   Test Accuracy: 94.30%
   Epoch 5, Loss: 0.2109, LR: 0.000010
   Test Accuracy: 94.50%
   ```
   实际值因随机初始化而异。

---

### 📖 关键点
- **调度器调用**：`scheduler.step()`通常在每个epoch末调用，更新学习率。
- **StepLR**：简单有效，每隔固定轮数降低学习率，适合快速实验。
- **学习率变化**：输出显示学习率从0.001降到0.00001，损失逐渐稳定。

---

### 📖 实际应用场景
- **深度学习**：学习率调度广泛用于CNN、RNN、Transformer（如BERT）。
- **复杂模型**：结合Adam优化器（如前述问题），调度器可提高收敛稳定性。
- **与其他正则化结合**：可与Dropout、BatchNorm、L2正则化等联合使用。

#### 注意事项
- **调度策略选择**：StepLR简单，余弦退火或ReduceLROnPlateau更灵活。
- **步长和衰减率**：`step_size`和`gamma`需根据任务调优。
- **调用顺序**：`scheduler.step()`通常在`optimizer.step()`后调用。
