## LSTM 
外部链接（Youtube）：[LSTM](https://www.youtube.com/watch?v=vTouAvxlphc)  
LSTM（长短期记忆网络，Long Short-Term Memory）  
- 重要性：
LSTM 是 RNN（循环神经网络）的一种变种，解决了标准 RNN 的梯度消失问题，能记住更长的序列信息。  
在语音识别、时间序列预测、文本生成等任务中广泛应用，是序列建模的经典模型。  
- 核心概念：
LSTM 通过“门机制”（输入门、遗忘门、输出门）控制信息的保留和遗忘，适合处理长序列数据。  
- 应用：语音助手（如 Siri）、股票价格预测、机器翻译。

<img width="969" height="304" alt="image" src="https://github.com/user-attachments/assets/1d0be4d9-f07c-4428-9aaa-cfe0ecbfd411" />  

- LSTM: 引入了门控机制来更好地处理长依赖关系。包括：      
   - Forget gate: 决定丢弃多少前一时刻信息。   
   - Input gate: 控制新输入x(t)的多少被加入。    
   - Output gate: 决定当前隐藏状态h(t)的输出。    
   - LSTM通过这些门控制信息流动，缓解梯度消失问题。   

## 代码 （Pytorch）
```
import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 下载并解压UCI HAR数据集
def download_and_extract_har():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "UCI_HAR_Dataset.zip"
    data_dir = "UCI HAR Dataset"
    if not os.path.exists(data_dir):
        print("正在下载UCI HAR数据集...")
        urllib.request.urlretrieve(url, zip_path)
        print("解压中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("数据集已准备好。")
    else:
        print("数据集已存在。")
    return data_dir

# 读取HAR数据集（这里只做二分类：WALKING vs. 其他）
def load_har_binary(data_dir):
    X_train = np.loadtxt(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt"))
    X_test = np.loadtxt(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt"))
    y_train = (y_train == 1).astype(np.float32)
    y_test = (y_test == 1).astype(np.float32)
    def reshape_X(X):
        X = X[:, :558]  # 9*62=558
        return X.reshape(X.shape[0], 9, 62)
    X_train = reshape_X(X_train)
    X_test = reshape_X(X_test)
    return X_train, y_train, X_test, y_test

# LSTM模型定义（同前）
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# 训练、测试、可视化函数同前
def train_model(train_data, train_labels, model, criterion, optimizer, batch_size=64, epochs=10):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        permutation = torch.randperm(train_data.size(0))
        epoch_loss = 0
        for i in range(0, train_data.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_data[indices].to(device), train_labels[indices].to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / train_data.size(0)
        loss_list.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    return loss_list

def batch_test(data, labels, model):
    model.eval()
    with torch.no_grad():
        inputs, targets = data.to(device), labels.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        accuracy = (preds == targets).float().mean().item()
        print(f'批量测试准确率: {accuracy*100:.2f}%')
        return preds.cpu().numpy(), targets.cpu().numpy()

def plot_loss_curve(loss_list):
    plt.figure(figsize=(8,4))
    plt.plot(loss_list, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 下载并加载数据
    data_dir = download_and_extract_har()
    X_train, y_train, X_test, y_test = load_har_binary(data_dir)
    print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
    # 转为torch张量
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # 参数
    input_size = train_data.shape[2]
    hidden_size = 32
    output_size = 1
    seq_length = train_data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练
    print("开始训练...")
    loss_list = train_model(train_data, train_labels, model, criterion, optimizer, batch_size=128, epochs=10)
    print("训练结束，开始批量测试...")
    batch_test(test_data, test_labels, model)
    plot_loss_curve(loss_list)
```
## 训练结果
开始训练...  
Epoch [2/10], Loss: 0.0628  
Epoch [4/10], Loss: 0.0118  
Epoch [6/10], Loss: 0.0154  
Epoch [8/10], Loss: 0.0014  
Epoch [10/10], Loss: 0.0003  
训练结束，开始批量测试...  
批量测试准确率: 98.71%  

<img width="785" height="388" alt="image" src="https://github.com/user-attachments/assets/983ef728-8e77-4cb7-9553-43494e967bda" /> 
图2 训练损失曲线

## 代码功能解释

代码实现了基于UCI HAR（人类活动识别）真实数据集的LSTM二分类模型训练与评估，其主要功能如下：  
- 1. 自动下载与解压真实数据  
自动从UCI官网下载“人类活动识别（HAR）”数据集并解压，无需手动准备数据。  
- 2. 数据预处理  
读取训练和测试集的特征（X_train, X_test）和标签（y_train, y_test）。
将原始每个样本的561维特征，裁剪为558维（9×62），重塑为9步序列，每步62维，适合LSTM输入。  
标签二值化：只区分WALKING（走路，标签1）和非WALKING（标签0），实现二分类。  
- 3. LSTM模型定义  
构建一个简单的LSTM神经网络，输入为序列数据（9步，每步62维），输出为二分类概率。  
LSTM输出最后一个时间步的隐藏状态，经全连接层和Sigmoid激活，得到分类概率。  
- 4. 训练与测试流程  
使用BCELoss（二分类交叉熵）和Adam优化器进行训练。  
支持批量训练和批量测试，自动计算并输出训练损失和测试准确率。  
训练过程中记录损失，便于后续可视化。  
- 5. 可视化 
绘制训练损失曲线，帮助观察模型收敛情况。 
### 整体功能总结  
这段代码实现了一个端到端的LSTM序列二分类流程，包括：  
自动下载和处理真实公开数据集  
数据预处理与序列化  
LSTM模型搭建  
训练、测试与准确率评估  
损失曲线可视化  
适用场景：任何需要用LSTM对真实时序数据进行二分类的任务，且可直接迁移到其它类似结构的数据集。

