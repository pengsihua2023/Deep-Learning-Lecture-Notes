## Python基础知识：Matplotlib库的基础知识
### 什么是 Matplotlib？

Matplotlib 是 Python 中一个强大的 2D（和部分 3D）绘图库，广泛用于数据可视化。它可以生成各种静态、动态和交互式图表，如折线图、散点图、柱状图、热图等，特别适合科学计算和数据分析。在深度学习中，Matplotlib 主要用于**可视化数据**、**模型性能**和**中间结果**，帮助理解数据分布、模型训练过程和调试。

#### Matplotlib 的核心特点：
- **灵活性**：支持多种图表类型和自定义样式。
- **与 NumPy/Pandas 集成**：直接处理 NumPy 数组或 Pandas DataFrame。
- **跨平台**：生成高质量图像，适用于论文、报告或 Web 应用。
- **生态丰富**：子模块（如 `pyplot`）简化绘图，Seaborn 等库基于 Matplotlib 提供高级接口。

在深度学习中，Matplotlib 常用于：
- 可视化数据集（如图像、特征分布）。
- 绘制训练过程中的损失曲线或准确率曲线。
- 显示模型预测结果（如混淆矩阵、特征图）。

---

### 在深度学习中需要掌握的 Matplotlib 知识

以下是深度学习中需要重点掌握的 Matplotlib 知识点，结合实际应用场景和代码示例。建议熟悉 `matplotlib.pyplot`（简称 `plt`），因为它是 Matplotlib 的主要接口。

#### 1. **基本绘图和设置**
   - **创建简单图表**：
     - 使用 `plt.plot()` 绘制折线图，`plt.scatter()` 绘制散点图。
       ```python
       import matplotlib.pyplot as plt
       import numpy as np
       x = np.linspace(0, 10, 100)
       y = np.sin(x)
       plt.plot(x, y, label='sin(x)')  # 折线图
       plt.legend()  # 显示图例
       plt.show()  # 显示图表
       ```
     - 深度学习场景：绘制训练和验证损失曲线。
   - **图表设置**：
     - 标题：`plt.title('Title')`
     - 轴标签：`plt.xlabel('X')`, `plt.ylabel('Y')`
     - 图例：`plt.legend()`
     - 网格：`plt.grid(True)`
       ```python
       plt.plot(x, y, label='sin(x)')
       plt.title('Sine Function')
       plt.xlabel('x')
       plt.ylabel('sin(x)')
       plt.grid(True)
       plt.legend()
       plt.show()
       ```
     - 深度学习场景：为损失曲线添加标题和标签，方便分析。

#### 2. **绘制多种图表类型**
   - **折线图**：
     - 用于显示连续变化，如损失或准确率随 epoch 变化。
       ```python
       epochs = np.arange(1, 11)
       train_loss = [0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
       val_loss = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
       plt.plot(epochs, train_loss, label='Train Loss')
       plt.plot(epochs, val_loss, label='Validation Loss')
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.legend()
       plt.show()
       ```
     - 深度学习场景：监控训练过程，判断过拟合或欠拟合。
   - **散点图**：
     - 显示数据点分布，如特征降维后的可视化（t-SNE/PCA 结果）。
       ```python
       x = np.random.randn(100)
       y = np.random.randn(100)
       plt.scatter(x, y, c='blue', alpha=0.5)
       plt.xlabel('Feature 1')
       plt.ylabel('Feature 2')
       plt.title('Feature Distribution')
       plt.show()
       ```
     - 深度学习场景：可视化分类任务的样本分布。
   - **柱状图**：
     - 显示类别统计，如每个类别的样本数量。
```
import matplotlib.pyplot as plt

classes = ['Class A', 'Class B', 'Class C']
counts = [50, 30, 20]
plt.bar(classes, counts, color='green')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()
```
<img width="590" height="456" alt="image" src="https://github.com/user-attachments/assets/6da66945-fa61-413e-b249-188c63dbc055" />

     - 深度学习场景：检查数据集是否平衡。
   - **直方图**：
     - 显示数据分布，如特征值或权重分布。
       ```python
       data = np.random.randn(1000)
       plt.hist(data, bins=30, color='purple', alpha=0.7)
       plt.xlabel('Value')
       plt.ylabel('Frequency')
       plt.title('Histogram of Data')
       plt.show()
       ```
     - 深度学习场景：检查数据归一化效果或模型权重分布。
   - **热图（Heatmap）**：
     - 显示矩阵数据，如混淆矩阵。
       ```python
       confusion_matrix = np.array([[50, 5, 2], [3, 45, 4], [1, 2, 48]])
       plt.imshow(confusion_matrix, cmap='Blues')
       plt.colorbar()
       plt.xlabel('Predicted')
       plt.ylabel('True')
       plt.title('Confusion Matrix')
       plt.show()
       ```
     - 深度学习场景：评估分类模型性能。

#### 3. **子图（Subplots）**
   - 使用 `plt.subplots()` 创建多个子图，适合同时显示多种信息。
     ```python
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1x2 子图
     ax1.plot(epochs, train_loss, label='Train Loss')
     ax1.set_title('Loss')
     ax1.legend()
     ax2.plot(epochs, [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92], label='Accuracy')
     ax2.set_title('Accuracy')
     ax2.legend()
     plt.tight_layout()  # 自动调整布局
     plt.show()
     ```
   - 深度学习场景：同时显示损失和准确率曲线，或比较多个模型。

#### 4. **图像显示**
   - 使用 `plt.imshow()` 显示图像数据，如输入图像或卷积层特征图。
     ```python
     image = np.random.rand(28, 28)  # 模拟 28x28 灰度图像
     plt.imshow(image, cmap='gray')
     plt.axis('off')  # 隐藏坐标轴
     plt.title('Sample Image')
     plt.show()
     ```
   - 深度学习场景：可视化 MNIST 图像或卷积神经网络的激活图。

#### 5. **自定义样式**
   - **颜色和线型**：
     - 颜色：`color='red'` 或 `c='r'`。
     - 线型：`linestyle='--'`（虚线）、`'-'`（实线）。
     - 标记：`marker='o'`（圆点）。
       ```python
       plt.plot(x, y, color='red', linestyle='--', marker='o', label='Data')
       plt.legend()
       plt.show()
       ```
   - **字体和大小**：
     - 设置全局样式：`plt.rcParams['font.size'] = 12`。
     - 单独设置：`plt.title('Title', fontsize=14)`.
   - **图表大小**：
     - `plt.figure(figsize=(8, 6))` 设置画布大小。
   - 深度学习场景：生成高质量图像，适合报告或论文。

#### 6. **保存图表**
   - 使用 `plt.savefig()` 保存为 PNG、JPG、PDF 等格式。
     ```python
     plt.plot(epochs, train_loss)
     plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')  # 高分辨率
     plt.show()
     ```
   - 深度学习场景：保存训练曲线或可视化结果，用于文档或演示。

#### 7. **与 Pandas/NumPy 集成**
   - **Pandas 绘图**：
     - Pandas 内置 Matplotlib 接口：`df.plot()`。
       ```python
       import pandas as pd
       df = pd.DataFrame({'epoch': epochs, 'loss': train_loss})
       df.plot(x='epoch', y='loss', title='Training Loss')
       plt.show()
       ```
     - 深度学习场景：快速可视化 Pandas 中的训练日志。
   - **NumPy 数组**：
     - Matplotlib 直接处理 NumPy 数组，如 `plt.plot(np.array([...]))`。
     - 深度学习场景：绘制模型输出的 NumPy 数组（如预测概率）。

#### 8. **交互式绘图（可选）**
   - 在 Jupyter Notebook 中启用交互模式：`%matplotlib notebook`。
   - 使用 `plt.ion()` 开启交互绘图，动态更新图表。
     ```python
     plt.ion()
     for i in range(10):
         plt.plot([i], [i**2], 'ro')
         plt.pause(0.1)
     ```
   - 深度学习场景：实时监控训练过程中的损失变化（通常用 TensorBoard 替代）。

#### 9. **高级可视化（视需求掌握）**
   - **3D 绘图**：
     - 使用 `mpl_toolkits.mplot3d` 绘制 3D 图。
       ```python
       from mpl_toolkits.mplot3d import Axes3D
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
       z = np.sin(np.sqrt(x**2 + y**2))
       ax.plot_surface(x, y, z, cmap='viridis')
       plt.show()
       ```
     - 深度学习场景：可视化高维特征空间（较少使用）。
   - **混淆矩阵（Seaborn 增强）**：
     - 结合 Seaborn（基于 Matplotlib）绘制更美观的热图。
       ```python
       import seaborn as sns
       sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
       plt.xlabel('Predicted')
       plt.ylabel('True')
       plt.show()
       ```

---

### 深度学习中的典型 Matplotlib 使用场景
1. **数据可视化**：
   - 绘制特征分布（直方图、散点图）。
   - 显示样本图像（如 MNIST、CIFAR-10）。
   - 检查类别分布（柱状图）。
2. **训练过程监控**：
   - 绘制损失曲线（训练和验证）。
   - 绘制准确率或其他指标曲线。
3. **模型评估**：
   - 显示混淆矩阵（热图）。
   - 绘制 ROC 曲线或 PR 曲线（结合 Scikit-learn）。
4. **调试**：
   - 可视化卷积层特征图。
   - 绘制权重或梯度分布。
5. **结果展示**：
   - 生成高质量图表，嵌入论文或报告。

---

### 需要掌握的 Matplotlib 核心功能总结
以下是深度学习中最常用的 Matplotlib 功能，建议熟练掌握：
- 基本绘图：`plt.plot`, `plt.scatter`, `plt.bar`, `plt.hist`, `plt.imshow`
- 图表设置：`plt.title`, `plt.xlabel`, `plt.ylabel`, `plt.legend`, `plt.grid`
- 子图：`plt.subplots`, `tight_layout`
- 保存：`plt.savefig`
- 样式：颜色、线型、标记、字体大小
- 与 Pandas/NumPy 集成：`df.plot`, NumPy 数组绘图
- 高级（可选）：热图（Seaborn）、3D 绘图

---

### 学习建议
- **实践**：用 Matplotlib 绘制真实深度学习任务的图表，如 MNIST 数据集的图像或训练损失曲线。
- **阅读文档**：Matplotlib 官方文档（matplotlib.org）提供教程和示例。
- **结合工具**：与 Pandas（数据处理）、NumPy（数值计算）、Seaborn（高级可视化）结合，完成数据分析和可视化任务。
- **项目驱动**：尝试用 Matplotlib 可视化 CNN 的特征图或分类模型的混淆矩阵。
- **参考 Seaborn**：Seaborn 提供更美观的默认样式，适合快速生成专业图表。

