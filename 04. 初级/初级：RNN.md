## 循环神经网络

循环神经网络（Recurrent Neural Network, RNN）  
- 重要性：RNN 适合处理序列数据（如文本、语音），是自然语言处理和时间序列预测的基础。
- 核心概念：
RNN 有“记忆”，可以记住之前的输入，适合处理有序数据（如句子）。  
变种如 LSTM（长短期记忆网络）能记住更长的序列。  
- 应用：语音识别（如 Siri）、机器翻译、股票预测。
- 为什么教：RNN 展示深度学习如何处理动态数据，与语音助手等应用相关。

<img width="956" height="304" alt="image" src="https://github.com/user-attachments/assets/ecdeb7fe-d4e1-4ef1-b7b9-9e766e6bf9bd" />  

RNN: 基本的循环神经网络单元，通过tanh激活函数处理输入Xt和前一时刻的隐藏状态h(t-1)，生成当前隐藏状态h(t)。它简单但容易遇到梯度消失问题，限制了长序列的处理能力。  
RNN（循环神经网络）被认为具有“记忆”是因为它通过隐藏状态h(t)在时间步之间传递信息。当前时刻的隐藏状态不仅依赖于当前输入x(t)，还依赖于前一时刻的隐藏状态h(t-1)，从而能够“记住”之前序列中的部分信息。这种结构使其适合处理序列数据，如时间序列或自然语言。然而，标准RNN的记忆能力有限，容易受梯度消失问题影响，难以捕捉长距离依赖。

## 1. RNN的基本结构
RNN是一种专门处理序列数据的神经网络，通过引入隐藏状态（hidden state）来捕获序列中的时间依赖关系。其核心思想是：当前时刻的输出不仅依赖当前输入，还依赖之前的隐藏状态。  
### 基本公式
对于时间步t，RNN的计算公式如下： 
隐藏状态更新  
 

---

$$
h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

* $x_t$：时间步 $t$ 的输入向量
* $h_t$：时间步 $t$ 的隐藏状态
* $h_{t-1}$：前一时间步的隐藏状态
* $W_{xh}$：输入到隐藏层的权重矩阵
* $W_{hh}$：隐藏层到隐藏层的权重矩阵
* $b_h$：隐藏层的偏置
* $\sigma$：激活函数（通常为 $\tanh$ 或 ReLU）

**输出：**

$$
y_t = W_{hy}h_t + b_y
$$

* $y_t$：时间步 $t$ 的输出
* $W_{hy}$：隐藏层到输出层的权重矩阵
* $b_y$：输出层的偏置

若需要非线性输出（如分类任务），可对 $y_t$ 施加激活函数（如 softmax）：

$$
o_t = \text{softmax}(y_t)
$$

---



## 2. 前向传播
<img width="856" height="274" alt="image" src="https://github.com/user-attachments/assets/6405f4f1-36d1-4306-a057-aef01620cd4a" />  


---

RNN 通过时间步迭代计算隐藏状态和输出。以序列 $x_1, x_2, \ldots, x_T$ 为例，前向传播过程如下：

1. 初始化 $h_0$（通常为零向量或随机初始化）。

2. 对每个时间步 $t = 1, 2, \ldots, T$：

   * 计算 $h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$
   * 计算 $y_t = W_{hy}h_t + b_y$

3. 根据任务需求，收集所有 $y_t$ 或仅使用最终输出 $y_T$。

---


## 3. 损失函数  
RNN通常使用损失函数来衡量预测输出与真实标签之间的差距。对于序列预测任务，常用交叉熵损失（分类）或均方误差（回归）。总损失为各时间步损失之和：  
<img width="613" height="152" alt="image" src="https://github.com/user-attachments/assets/fc7b1b5a-b37a-4ae3-aefd-7aefe3f259bd" />  

---

$$
L = \sum_{t=1}^{T} L_t(\hat{y}_t, y_t)
$$

其中 $L_t$ 为时间步 $t$ 的损失，$\hat{y}_t$ 为预测输出，$y_t$ 为真实标签。

---

## 4. 反向传播（Backpropagation Through Time, BPTT） 
<img width="1003" height="752" alt="image" src="https://github.com/user-attachments/assets/4d71329f-a372-476f-bf17-f6b094056b40" />
---

RNN 的训练通过反向传播算法沿时间步展开，称为 **BPTT**。目标是最小化损失函数 $L$，通过梯度下降更新权重
$W_{xh}, W_{hh}, W_{hy}$ 和偏置 $b_h, b_y$。

### 梯度计算

* **对每一时间步 $t$，计算损失对隐藏状态的梯度：**

$$
\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L_{t+1}}{\partial h_t} + \cdots + \frac{\partial L_T}{\partial h_t}
$$

其中， $\frac{\partial L_{t+k}}{\partial h_t}$ 通过链式法则沿时间步递归计算。

* **权重梯度：**

$$
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{xh}}
$$

类似地计算 $W_{hh}, W_{hy}, b_h, b_y$ 的梯度。

---

### 梯度消失 / 爆炸问题

由于 $h_t$ 依赖 $h_{t-1}$，梯度通过矩阵 $W_{hh}$ 的多次乘法传递，可能导致：

* **梯度消失**：梯度过小，早期时间步的影响难以传递。
* **梯度爆炸**：梯度过大，训练不稳定。

---

### 解决方法

* **梯度裁剪**（限制梯度大小）；
* **更先进的结构**：如 LSTM 和 GRU。

---






## 代码（Pytorch）
```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 显示负号

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成简单序列数据
def generate_sequence_data(num_samples, seq_length):
    data = []
    labels = []
    for _ in range(num_samples):
        # 随机生成序列（0或1）
        seq = np.random.randint(0, 2, size=(seq_length,))
        # 标签：序列中1的数量是否超过一半
        label = 1 if np.sum(seq) > seq_length // 2 else 0
        data.append(seq)
        labels.append(label)
    return torch.FloatTensor(data).unsqueeze(-1), torch.LongTensor(labels)

# 数据参数
num_samples = 1000
seq_length = 10
input_size = 1
hidden_size = 16
num_classes = 2

# 生成训练和测试数据
train_data, train_labels = generate_sequence_data(num_samples, seq_length)
test_data, test_labels = generate_sequence_data(num_samples // 5, seq_length)

# 2. 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        # RNN前向传播
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 3. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
def train_model(num_epochs=20):
    model.train()
    loss_list = []  # 记录每个epoch的损失
    for epoch in range(num_epochs):
        inputs, labels = train_data.to(device), train_labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())  # 保存损失
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return loss_list  # 返回损失列表

# 5. 测试模型
def test_model():
    model.eval()
    with torch.no_grad():
        inputs, labels = test_data.to(device), test_labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return predicted.cpu().numpy(), labels.cpu().numpy()  # 返回预测和真实标签

# 6. 可视化函数
def plot_loss_curve(loss_list):
    """绘制训练损失曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(pred, true):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.show()

# 7. 执行训练和测试
if __name__ == "__main__":
    print("Training started...")
    loss_list = train_model(num_epochs=20)
    print("\nTesting started...")
    pred, true = test_model()
    # 可视化
    plot_loss_curve(loss_list)
    plot_confusion_matrix(pred, true)
```
## 训练结果
Training started...
Epoch [5/20], Loss: 0.6591  
Epoch [10/20], Loss: 0.6286  
Epoch [15/20], Loss: 0.5319  
Epoch [20/20], Loss: 0.3664  

Testing started...  
Test Accuracy: 79.00%  

<img width="791" height="385" alt="image" src="https://github.com/user-attachments/assets/ea2bbd16-e1b1-4334-a856-d1e91223a8a7" /> 
图2 训练损失曲线

<img width="396" height="295" alt="image" src="https://github.com/user-attachments/assets/4c5c3a4d-986f-4b42-8612-9a05ed4e3f87" />  

图3 混淆矩阵

## 代码功能简述
这段代码实现的功能简述如下：
- 1. 功能概述
该代码实现了一个基于PyTorch的简单RNN序列二分类模型，并包含了训练过程损失曲线和测试集混淆矩阵的可视化。主要流程包括数据生成、模型定义、训练、测试和可视化。
- 2. 主要步骤说明
   - （1）数据生成
随机生成二值序列（0或1），每个序列长度为10。
标签规则：如果序列中1的数量超过一半，则标签为1，否则为0。
生成1000个训练样本和200个测试样本。
   - （2）模型定义
构建了一个简单的RNN模型（SimpleRNN），输入为序列数据，输出为二分类的概率分布。
RNN输出最后一个时间步的隐藏状态，经全连接层映射为2类。
   - （3）训练过程
使用交叉熵损失函数和Adam优化器。
训练20个epoch，记录每个epoch的损失，并每5个epoch打印一次损失。
   - （4）测试与评估
在测试集上评估模型，输出准确率。
返回预测标签和真实标签。
   - （5）可视化
绘制训练损失曲线，直观展示模型收敛过程。
绘制混淆矩阵，展示模型在测试集上的分类效果（正确/错误分类数量）。
- 3. 适用场景
适合用于序列二分类任务的入门演示，如简单的时序信号、文本、事件流等。
可作为RNN模型结构、训练与评估流程的模板。
- 4. 总结
本代码实现了一个端到端的RNN序列二分类任务，包括数据生成、模型训练、测试评估和可视化，适合深度学习初学者理解RNN的基本用法和分类流程。
