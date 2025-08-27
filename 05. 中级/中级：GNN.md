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
