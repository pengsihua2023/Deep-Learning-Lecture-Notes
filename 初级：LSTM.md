## LSTM 
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
   - Input gate: 控制新输入$x_t$的多少被加入。    
   - Output gate: 决定当前隐藏状态$h_t$的输出。    
   - LSTM通过这些门控制信息流动，缓解梯度消失问题。   

## 代码
```
import torch
import torch.nn as nn
import torch.optim as optim

# 定义LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size)
        c0 = torch.zeros(1, batch_size, self.hidden_size)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# 参数设置
input_size = 10   # 每个时间步的特征维度
hidden_size = 20  # 隐藏层维度
output_size = 1   # 输出维度（二分类）
seq_length = 5    # 序列长度
batch_size = 32
epochs = 10
learning_rate = 0.01

# 模型、损失函数和优化器
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模拟数据
data = torch.randn(batch_size, seq_length, input_size)
labels = torch.randint(0, 2, (batch_size, 1)).float()

# 训练循环
for epoch in range(epochs):
    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, labels)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_input = torch.randn(1, seq_length, input_size)
    prediction = model(test_input)
    print(f"Input shape: {test_input.shape}, Prediction: {prediction.item():.4f}")
```
