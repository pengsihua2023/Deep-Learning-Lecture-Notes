## Autoencoder 
Autoencoder（自编码器）
- 重要性：
Autoencoder 是一种无监督学习模型，用于数据压缩、降噪或特征学习。  
它是生成模型（如 GAN）的先驱，广泛用于数据预处理和异常检测。  
- 核心概念：
Autoencoder 包含编码器（压缩数据）和解码器（重构数据），目标是让输出尽可能接近输入。  
比喻：像一个“数据压缩机”，把大文件压缩后再解压，尽量保持原样。  
- 应用：图像去噪、数据压缩、异常检测（如信用卡欺诈检测）。

## 代码
```
import torch
import torch.nn as nn
import torch.optim as optim

# 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 参数设置
input_dim = 784  # 假设输入是28x28的图像展平后的维度 (如MNIST)
hidden_dim = 128  # 隐层维度
learning_rate = 0.001
epochs = 10
batch_size = 32

# 模型、损失函数和优化器
model = Autoencoder(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模拟数据 (这里用随机数据代替，实际使用时需替换为真实数据集)
data = torch.randn(1000, input_dim)

# 训练循环
for epoch in range(epochs):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        # 前向传播
        output = model(batch)
        loss = criterion(output, batch)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    reconstructed = model(test_input)
    print(f"Input shape: {test_input.shape}, Reconstructed shape: {reconstructed.shape}")

```
## 运行结果
Epoch [7/10], Loss: 0.9657  
Epoch [8/10], Loss: 0.9408  
Epoch [9/10], Loss: 0.9078  
Epoch [10/10], Loss: 0.8681  
Input shape: torch.Size([1, 784]), Reconstructed shape: torch.Size([1, 784])  
