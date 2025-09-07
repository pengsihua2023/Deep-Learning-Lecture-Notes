# 多任务学习（Multi-Task Learning, MTL）

<div align="center">
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/4dd18183-6e9e-4418-ab2b-b0f9e8edb4bb" />
</div>
<div align="center">
（此图引自Internet）
</div>

## 📖 定义  
多任务学习（Multi-Task Learning, MTL）是一种机器学习训练范式，核心思想是：一个模型同时学习多个相关任务，而不是像传统方法那样为每个任务单独训练模型。模型共享大部分参数，每个任务有特定输出头，联合优化多个目标。  


## 📖 多任务学习的数学描述

### 1. 单任务学习的基本形式

给定数据集：

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N,
$$

* $x_i \in \mathcal{X}$：第 $i$ 个样本的输入特征。  
* $y_i \in \mathcal{Y}$：第 $i$ 个样本对应的监督信号（标签）。  
* $N$：训练样本数量。  

我们训练一个参数为 $\theta$ 的模型：

$$
f_\theta : \mathcal{X} \to \mathcal{Y},
$$

目标是最小化期望损失：

$$
\min_\theta \ \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), y) \right].
$$



### 2. 多任务学习的扩展形式

假设有 $T$ 个任务，每个任务 $t$ 的数据集为：

$$
\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t},
$$

* $x_i^t$：任务 $t$ 的输入。  
* $y_i^t$：任务 $t$ 的标签。  
* $N_t$：任务 $t$ 的样本数量。  

每个任务对应损失函数 $\mathcal{L}_t$。多任务学习优化目标是：

$$
\min_\theta \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \Big[ \mathcal{L}_t(f_\theta(x), y) \Big].
$$

* $\lambda_t$：任务权重，控制不同任务在整体目标中的重要性。  



### 3. 参数共享的结构化表示

实际中常用 **共享表示层 + 任务专用输出层**：

1. **共享表示层**：

$$
h = \phi_{\theta_s}(x),
$$

* $\phi_{\theta_s}$：特征抽取器（如神经网络的前几层），参数 $\theta_s$ 在所有任务中共享。  
* $h$：共享的隐含表示（latent representation）。  

2. **任务专用输出层**：

$$
\hat{y}^t = f^t_{\theta_t}(h),
$$

* $f^t_{\theta_t}$：任务 $t$ 的预测器，参数 $\theta_t$ 仅供任务 $t$ 使用。  
* $\hat{y}^t$：模型对任务 $t$ 的预测。  

整体优化目标：

$$
\min_{\theta_s, \{\theta_t\}_{t=1}^T} \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \left[ \mathcal{L}_t(f^t_{\theta_t}(\phi_{\theta_s}(x)), y) \right].
$$



### 4. 矩阵/正则化视角

若假设任务参数矩阵为：

$$
W = [\theta_1, \dots, \theta_T] \in \mathbb{R}^{d \times T},
$$

则可在损失函数外加正则化约束：

### (a) 低秩约束

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \lambda \|W\|_*
$$

* $\|W\|_*$：核范数，促使 $W$ 的秩较低，表示任务共享一个低维子空间。  

### (b) 图正则化

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \gamma \sum_{(i,j)\in E} \|W_i - W_j\|^2
$$

* $E$：任务关系图的边集合。  
* $\|W_i - W_j\|^2$：鼓励相似任务的参数接近。  



### 5. 贝叶斯视角

引入任务参数的先验分布：

$$
p(\theta_1, \dots, \theta_T | \alpha) = \prod_{t=1}^T p(\theta_t | \alpha)
$$

* $\alpha$：共享的超参数，控制所有任务的先验分布。  



### 总结

多任务学习的数学建模有三种主要思路：

1. **加权损失函数**（任务简单相加，带权重 $\lambda_t$）；  
2. **参数共享**（共享层 $\theta_s$ + 任务专用头 $\theta_t$）；  
3. **正则化 / 概率建模**（通过核范数、图正则化或共享先验建模任务关系）。  
---
## 📖 Code
一个基于PyTorch的最简单Multi-Task Learning（MTL）示例，使用真实数据集（UCI Wine Quality数据集），实现两个任务：预测葡萄酒质量（回归任务）和预测葡萄酒是否优质（分类任务，质量≥6为优质）。结果将通过可视化（预测质量的散点图）和评估指标（回归的MSE、分类的准确率）来展示。


```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 定义多任务学习模型
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.classification_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        quality_pred = self.regression_head(shared_features)
        is_good_pred = self.classification_head(shared_features)
        return quality_pred, is_good_pred

# 数据准备
def prepare_data():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    X = data.drop('quality', axis=1).values
    y_quality = data['quality'].values
    y_class = (y_quality >= 6).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = train_test_split(
        X, y_quality, y_class, test_size=0.2, random_state=42
    )
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_quality_train = torch.FloatTensor(y_quality_train).reshape(-1, 1)
    y_quality_test = torch.FloatTensor(y_quality_test).reshape(-1, 1)
    y_class_train = torch.FloatTensor(y_class_train).reshape(-1, 1)
    y_class_test = torch.FloatTensor(y_class_test).reshape(-1, 1)
    
    return X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test

# 训练模型
def train_model(model, X_train, y_quality_train, y_class_train, epochs=100, lr=0.01):
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        quality_pred, is_good_pred = model(X_train)
        loss_reg = criterion_reg(quality_pred, y_quality_train)
        loss_cls = criterion_cls(is_good_pred, y_class_train)
        loss = loss_reg + loss_cls
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Regression Loss: {loss_reg.item():.4f}, Classification Loss: {loss_cls.item():.4f}')

# 评估和可视化
def evaluate_and_visualize(model, X_test, y_quality_test, y_class_test):
    model.eval()
    with torch.no_grad():
        quality_pred, is_good_pred = model(X_test)
        quality_pred = quality_pred.numpy()
        is_good_pred = (torch.sigmoid(is_good_pred) > 0.5).float().numpy()
        y_quality_test = y_quality_test.numpy()
        y_class_test = y_class_test.numpy()
    
    mse = mean_squared_error(y_quality_test, quality_pred)
    accuracy = accuracy_score(y_class_test, is_good_pred)
    print(f'\nTest Set Evaluation:')
    print(f'Regression MSE: {mse:.4f}')
    print(f'Classification Accuracy: {accuracy:.4f}')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_quality_test, quality_pred, alpha=0.5)
    plt.plot([y_quality_test.min(), y_quality_test.max()], [y_quality_test.min(), y_quality_test.max()], 'r--')
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title('Wine Quality Prediction (Regression Task)')
    plt.tight_layout()
    plt.savefig('wine_quality_prediction.png')
    plt.close()
    print("Prediction scatter plot saved as 'wine_quality_prediction.png'")

    print("\nSample Predictions (First 5):")
    for i in range(5):
        print(f"Sample {i+1}: True Quality={y_quality_test[i][0]:.2f}, Predicted Quality={quality_pred[i][0]:.2f}, "
              f"True Class={y_class_test[i][0]:.0f}, Predicted Class={is_good_pred[i][0]:.0f}")

def main():
    X_train, X_test, y_quality_train, y_quality_test, y_class_train, y_class_test = prepare_data()
    model = MultiTaskModel(input_dim=11, hidden_dim=64)
    train_model(model, X_train, y_quality_train, y_class_train, epochs=100)
    evaluate_and_visualize(model, X_test, y_quality_test, y_class_test)

if __name__ == "__main__":
    main()
````


## 📖 代码说明：

1. **数据集**：

   * 使用 UCI Wine Quality 数据集（红酒，1599 条样本），包含 11 个化学特征和质量评分（3-8 分）。
   * 任务1（回归）：预测质量分数。
   * 任务2（分类）：预测是否优质（质量 ≥ 6）。
   * 数据通过 `pandas` 从 UCI 网站加载，标准化后划分为训练集（80%）和测试集（20%）。

2. **模型结构**：

   * 共享层：两层全连接（ReLU 激活），输入 11 维特征，隐层 64 维。
   * 回归头：输出 1 维质量分数。
   * 分类头：输出 1 维二分类概率（优质/非优质）。
   * 损失函数：回归用 `MSELoss`，分类用 `BCEWithLogitsLoss`，联合损失为两者之和。

3. **训练**：

   * 使用 Adam 优化器，学习率 0.01，训练 100 个 epoch。
   * 每 20 个 epoch 打印总损失、回归损失和分类损失。

4. **评估与可视化**：

   * 评估回归任务的均方误差（MSE）和分类任务的准确率。
   * 绘制散点图，展示真实质量与预测质量的关系，保存为 `wine_quality_prediction.png`。
   * 打印前 5 个测试样本的真实和预测值（质量分数和分类结果）。

5. **依赖**：

   * 需安装 `torch`、`sklearn`、`pandas`、`matplotlib`、`seaborn`

     ```bash
     pip install torch scikit-learn pandas matplotlib seaborn datasets
     ```
   * 数据集在线加载，无需手动下载。


## 📖 运行结果：

* 输出训练过程中的损失值。
* 测试集评估：

  * 回归任务的 MSE（反映预测质量分数的误差）。
  * 分类任务的准确率（反映优质/非优质分类正确率）。
* 生成 `wine_quality_prediction.png`，展示预测质量与真实质量的散点图（红线为理想预测线）。
* 打印前 5 个样本的预测结果，展示真实和预测的质量分数及分类结果。


