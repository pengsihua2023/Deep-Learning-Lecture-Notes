
👌 我来帮你检查并修正 LaTeX 格式问题（主要是括号、空格和大括号使用不规范导致的）。我会给出一份**排版正确**、可以直接放进论文或笔记的版本。

---

# 图神经网络的数学化定义（修正版）

## 1. 图的基本结构

一个图定义为三元组：

$$
G = (V, E, X)
$$

其中：

* $V = \{1, 2, \dots, N\}$ 为节点集合，节点数为 $N$。
* $E \subseteq V \times V$ 为边集合。
* $X \in \mathbb{R}^{N \times d}$ 为节点特征矩阵，其中第 $i$ 行 $x_i \in \mathbb{R}^d$ 是节点 $i$ 的初始特征。

若采用邻接矩阵表示，则 $A \in \mathbb{R}^{N \times N}$，其中 $A_{ij} \neq 0$ 表示 $(i,j) \in E$。

---

## 2. 节点表示的迭代更新

GNN 的核心思想是 **消息传递 (Message Passing)**。在第 $k$ 层，每个节点 $i$ 的表示由自己和邻居的上一层表示决定：

$$
h_i^{(k)} = \psi^{(k)}\!\left(h_i^{(k-1)},\;\phi^{(k)}\!\left(\{h_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right), 
\quad h_i^{(0)} = x_i
$$

其中：

* $\mathcal{N}(i)$ 为节点 $i$ 的邻居集合（可包含自己）。
* $\phi^{(k)}: \mathcal{P}(\mathbb{R}^{d_{k-1}}) \to \mathbb{R}^{d_{k-1}}$ 是 **聚合函数 (aggregation)**，对邻居节点嵌入进行汇总。
* $\psi^{(k)}: \mathbb{R}^{d_{k-1}} \times \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{d_k}$ 是 **更新函数 (update)**，结合节点自身和邻居信息生成新的表示。
* 经过 $K$ 层传播后，得到节点嵌入：

  $$
  H^{(K)} = \{h_i^{(K)}\}_{i=1}^N
  $$

---

## 3. 图级表示

若任务需要对整个图进行预测（如图分类），则在最后一层节点表示的基础上定义图表示：

$$
h_G = \rho\!\left(\{h_i^{(K)} : i \in V\}\right)
$$

其中 $\rho: \mathcal{P}(\mathbb{R}^{d_K}) \to \mathbb{R}^{d_G}$ 是 **读出函数 (readout)**，常见形式包括 sum、mean、max pooling 或基于注意力的加权和。

---

## 4. 特例：常见 GNN 实现

* **GCN (Graph Convolutional Network)**

  $$
  H^{(k)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(k-1)} W^{(k)}\right)
  $$

  其中 $\tilde{A} = A + I$，$\tilde{D}$ 是 $\tilde{A}$ 的度矩阵。

* **GraphSAGE**

  $$
  h_i^{(k)} = \sigma\!\left(W^{(k)} \cdot \mathrm{concat}\!\left(h_i^{(k-1)},\;\phi^{(k)}\!\left(\{h_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right)\right)
  $$

* **GAT (Graph Attention Network)**

  $$
  h_i^{(k)} = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j^{(k-1)}\right)
  $$

  其中 $\alpha_{ij}^{(k)}$ 是注意力权重。

---

## 5. 总结

一个 GNN 的数学定义可以概括为：

1. **输入**：图 $G=(V,E,X)$。
2. **传播规则**：

   $$
   h_i^{(k)} = \psi^{(k)}\!\left(h_i^{(k-1)},\;\phi^{(k)}\!\left(\{h_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right)
   $$
3. **输出**：节点表示 $H^{(K)}$ 或图表示 $h_G$。

---

✅ 这份修正版解决了：

* 多余分号的问题。
* 括号使用不统一的问题（现在用 $\left(\cdot\right)$、$\!$ 来保证紧凑美观）。
* GraphSAGE 中的 `concat` 改成 `\mathrm{concat}`，避免和变量混淆。

---

要不要我再帮你做一个 **统一对照表（$\phi, \psi, \rho$ 的含义总结表格）**，方便放进附录？



# 图神经网络 (GNN) 数学定义对照表

| 符号                                                                                  | 定义                                                                                                               | 说明                                                           |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| $G = (V,E,X)$                                                                       | 图结构                                                                                                              | $V$ 为节点集合，$E$ 为边集合，$X \in \mathbb{R}^{N \times d}$ 为初始节点特征矩阵 |
| $h_i^{(0)} = x_i$                                                                   | 节点初始表示                                                                                                           | 节点 $i$ 的特征向量                                                 |
| $\mathcal{N}(i)$                                                                    | 节点邻居集合                                                                                                           | 与节点 $i$ 相连的所有节点（可含自身）                                        |
| $\phi^{(k)}: \mathcal{P}(\mathbb{R}^{d_{k-1}}) \to \mathbb{R}^{d_{k-1}}$            | 聚合函数 (Aggregation)                                                                                               | 从邻居节点嵌入集合中提取信息，例如 sum、mean、max、attention                     |
| $\psi^{(k)}: \mathbb{R}^{d_{k-1}} \times \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{d_k}$ | 更新函数 (Update)                                                                                                    | 将节点自身表示与邻居聚合结果结合，通常是 MLP                                     |
| 节点更新规则                                                                              | $\displaystyle h_i^{(k)} = \psi^{(k)}\Big(h_i^{(k-1)}, \;\phi^{(k)}(\{h_j^{(k-1)}: j \in \mathcal{N}(i)\})\Big)$ | **消息传递公式**：第 $k$ 层节点表示由自身和邻居共同决定                             |
| $H^{(K)} = \{h_i^{(K)}\}_{i=1}^N$                                                   | 节点最终表示                                                                                                           | 经过 $K$ 层传播后的节点嵌入矩阵                                           |
| $\rho: \mathcal{P}(\mathbb{R}^{d_K}) \to \mathbb{R}^{d_G}$                          | 读出函数 (Readout)                                                                                                   | 将所有节点嵌入映射为图级表示，常用 sum/mean/max pooling 或注意力                  |
| 图表示                                                                                 | $\displaystyle h_G = \rho(\{h_i^{(K)}: i \in V\})$                                                               | 得到整个图的全局表示，用于图分类等任务                                          |

---

✨ 这样，一个 GNN 的完整数学定义可以总结为：

1. **输入**：图 $G=(V,E,X)$
2. **传播**：消息传递迭代

   $   h_i^{(k)} = \psi^{(k)}\Big(h_i^{(k-1)}, \;\phi^{(k)}(\{h_j^{(k-1)}: j \in \mathcal{N}(i)\})\Big)   $
3. **输出**：节点嵌入 $H^{(K)}$ 或图嵌入 $h_G$。

---
你观察得很仔细 👍。
我前面写的公式里多了一个 **分号**，严格来说不合适。

正确的 **消息传递形式**应该是：

$$
h_i^{(k)} = \psi^{(k)}\!\left( h_i^{(k-1)},  \phi^{(k)}\{\big(\{h_j^{(k-1)} : j \in \mathcal{N}(i)\}\}\big) \right)
$$

---

### 解释

* $\phi^{(k)}(\cdot)$ → 聚合函数 (aggregation)，输入是邻居节点表示的集合。
* $\psi^{(k)}(\cdot, \cdot)$ → 更新函数 (update)，输入是**自己上一层的表示**和**聚合结果**两个参数。
* 初始条件：

  $$
  h_i^{(0)} = x_i
  $$
* 最终节点表示：

  $$
  H^{(K)} = \{h_i^{(K)}\}_{i=1}^N
  $$
* 若任务是图分类，还需读出函数：

  $$
  \h_G = \rho\big(\{h_i^{(K)} : i \in V\}\big)
  $$

---

要不要我帮你把 **GCN / GraphSAGE / GAT** 都用这个更严谨的 $\psi,\phi$ 形式统一改写？


## GNN
<div align="center">
<img width="500" height="263" alt="image" src="https://github.com/user-attachments/assets/47f67caf-be26-42b4-928e-b8db05f1afab" />  
</div>

Graph Neural Network (GNN, 图神经网络)  
- 重要性：
GNN 专门处理图结构数据（如社交网络、分子结构），在推荐系统、化学建模和知识图谱中应用广泛。  
它是深度学习向非欧几里得数据（如图、网络）扩展的关键，代表了现代 AI 的前沿方向。  
- 核心概念：
图由节点（点）和边（连接）组成，GNN 通过“消息传递”让节点聚合邻居信息，学习图的结构和特征。  
比喻：像“朋友圈信息传播”，每个节点（人）根据朋友的信息更新自己的状态。  
- 应用：推荐系统（如 Netflix 推荐）、分子设计（药物发现）、交通网络分析。



编写一个基于PyTorch和PyTorch Geometric的最简单Graph Neural Network（GNN）示例，使用真实数据集（Cora数据集，常用的图分类基准数据集），实现节点分类任务。模型使用简单的Graph Convolutional Network（GCN）。结果将通过可视化节点嵌入（t-SNE降维）和评估分类准确率来展示。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 定义简单的GCN模型
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 可视化节点嵌入
def visualize_embeddings(embeddings, labels, num_classes, title="t-SNE Visualization of Node Embeddings"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {i}', alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig('cora_embeddings.png')
    plt.close()
    print("t-SNE visualization saved as 'cora_embeddings.png'")

# 训练和评估
def train_and_evaluate(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {acc:.4f}')
            model.train()
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        print(f'\nTest Accuracy: {test_acc:.4f}')
        
        # 获取嵌入（最后一层输出）
        embeddings = out.cpu().numpy()
        labels = data.y.cpu().numpy()
        visualize_embeddings(embeddings, labels, num_classes=data.num_classes)

def main():
    # 加载Cora数据集
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    data = data.to(device)
    
    # 初始化模型
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes).to(device)
    
    # 训练和评估
    train_and_evaluate(model, data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
```

### 代码说明：
1. **数据集**：
   - 使用Cora数据集（2708个节点，7个类别，1433维特征，表示学术论文及其引用关系）。
   - 每个节点是论文，特征是词袋表示，边是引用关系，任务是预测论文类别。
   - 数据通过`torch_geometric`的`Planetoid`加载，包含训练、验证和测试掩码。

2. **模型结构**：
   - 简单GCN：两层GCNConv（图卷积层），第一层将1433维特征映射到16维，第二层映射到7维（类别数）。
   - 使用ReLU激活和Dropout（p=0.5）防止过拟合。

3. **训练**：
   - 使用Adam优化器，学习率0.01，权重衰减5e-4，训练200个epoch。
   - 损失函数为交叉熵，仅对训练掩码的节点计算损失。
   - 每50个epoch打印训练损失和验证集准确率。

4. **评估与可视化**：
   - **评估**：在测试集上计算节点分类准确率。
   - **可视化**：对模型输出的节点嵌入（最后一层输出）使用t-SNE降维到2D，绘制散点图，按类别着色，保存为`cora_embeddings.png`。
   - 理想情况下，同一类别的节点在嵌入空间中应聚类。

5. **依赖**：
   - 需安装`torch`、`torch_geometric`、`sklearn`、`matplotlib`（`pip install torch torch-geometric scikit-learn matplotlib`）。
   - Cora数据集会自动下载到`./data`目录。

### 运行结果：
- 输出每50个epoch的训练损失和验证准确率。
- 输出测试集的最终分类准确率。
- 生成`cora_embeddings.png`，展示节点嵌入的2D分布，颜色表示不同类别。
- 散点图反映GNN是否学习到有意义的嵌入（同类节点应靠近，异类节点应分开）。

### 注意：
- 散点图保存在运行目录下，可用图像查看器检查。
- 模型简单（两层GCN），适合展示GNN概念；实际应用可增加层数或使用更复杂的GNN变体（如GAT）。
