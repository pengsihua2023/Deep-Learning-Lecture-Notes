## 初级：GRU
<img width="552" height="277" alt="image" src="https://github.com/user-attachments/assets/2c7f3eef-f4be-471c-b7df-cd62b479df28" />

<img width="884" height="266" alt="image" src="https://github.com/user-attachments/assets/75186129-08a6-478c-b91e-82a65e0a601f" />  

门控循环单元（Gated Recurrent Unit, GRU）是一种常用于处理序列数据的循环神经网络（RNN）变体，由Kyunghyun Cho等人于2014年提出。GRU旨在解决传统RNN在长序列处理中遇到的梯度消失或爆炸问题，同时简化了长短期记忆网络（LSTM）的结构，具有更低的计算复杂度和更少的参数。  
### GRU的核心思想
GRU通过引入更新门（update gate）和重置门（reset gate）来控制信息的流动和遗忘，从而有效捕捉序列中的长期依赖关系。与LSTM相比，GRU将遗忘门和输入门合并为单一的更新门，简化了结构，但仍然保留了强大的建模能力。  
### GRU的工作机制
<img width="1229" height="866" alt="image" src="https://github.com/user-attachments/assets/0c326801-56ce-4125-adec-ca7403e67332" />  

### GRU的特点
- 简化结构：相比LSTM，GRU只有两个门（更新门和重置门），参数更少，计算效率更高。
- 长期依赖：通过门控机制，GRU能有效捕捉长序列中的依赖关系，缓解梯度消失问题。
- 灵活性：GRU适用于多种序列建模任务，如自然语言处理（NLP）、时间序列预测等。

### GRU与LSTM的对比
-相似点：两者都通过门控机制解决RNN的梯度问题，适合长序列任务。
- 不同点：
GRU结构更简单，参数更少，训练速度更快。  
LSTM有独立的记忆单元，适合更复杂的任务，但计算成本较高。  
在实际应用中，GRU和LSTM性能因任务而异，需根据具体场景选择。  
### 应用场景
- GRU广泛应用于：
自然语言处理：如机器翻译、文本生成、情感分析。  
时间序列分析：如股票预测、天气预测。  
语音处理：如语音识别、语音合成。  

### 总结
GRU是一种高效、简化的循环神经网络变体，通过更新门和重置门实现信息的选择性传递和遗忘。它在保持强大序列建模能力的同时，降低了计算复杂度，是许多序列任务的理想选择。  
## 实例
提供一个使用Python、PyTorch和真实数据集（正弦波序列）的简单GRU示例，展示其原理，并通过Matplotlib可视化预测结果。示例使用正弦波数据进行序列预测，GRU学习序列模式并预测后续值。代码包括数据准备、GRU模型定义、训练和可视化。  

### 说明

数据集：使用正弦波（sin(t)）作为真实数据，生成1000个点。每个样本包含10个连续点作为输入，预测下一个点。  
- GRU模型：  
输入尺寸为1（单变量时间序列）。  
隐藏层尺寸为16（简单网络，足以捕捉正弦波模式）。  
输出尺寸为1（预测下一个值）。 
GRU层处理序列，线性层（fc）将最后一个时间步的输出映射到预测值。  

- 训练：使用Adam优化器和均方误差（MSE）损失函数，训练100个epoch。
可视化：通过Matplotlib绘制真实值（蓝色实线）和预测值（红色虚线），展示GRU对正弦波模式的拟合效果。
## 代码
```
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成正弦波数据
t = np.linspace(0, 20, 1000)  # 时间轴
data = np.sin(t)  # 正弦波
sequence_length = 10  # 序列长度
X, y = [], []

# 准备输入-输出对：用前10个点预测下一个点
for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])
X = np.array(X).reshape(-1, sequence_length, 1)  # [样本数, 序列长度, 特征数]
y = np.array(y).reshape(-1, 1)  # [样本数, 1]
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# 2. 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # GRU前向传播
        out, _ = self.gru(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 3. 训练模型
model = GRUModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 4. 预测
model.eval()
with torch.no_grad():
    pred = model(X).numpy()

# 5. 可视化
plt.figure(figsize=(10, 5))
plt.plot(t[sequence_length:], y.numpy(), label='True Values', color='blue')
plt.plot(t[sequence_length:], pred, label='Predicted Values', color='red', linestyle='--')
plt.title('GRU Prediction on Sine Wave')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```
### 结果解读

GRU通过更新门和重置门学习正弦波的周期性模式。  
图表中，预测值应接近真实值，表明GRU能有效捕捉序列的规律。  
若预测偏差较大，可增加hidden_size或epochs，或调整lr。  
