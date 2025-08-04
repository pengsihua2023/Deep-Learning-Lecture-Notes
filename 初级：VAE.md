## VAE（变分自编码器，Variational Autoencoder）
- 重要性：
VAE 是 Autoencoder 的生成式扩展，结合概率模型，能生成新数据（如图片、文本）。  
在生成模型领域与 GAN 齐名，适合数据生成和分布学习。 
- 核心概念：
VAE 将编码器输出映射到一个概率分布（通常是正态分布），解码器从分布中采样生成数据。  
- 应用：生成艺术、数据增强、异常检测。

<center><img width="617" height="376" alt="image" src="https://github.com/user-attachments/assets/d8b5e82e-5b83-41d9-8b3c-521a3aeeb38e" /></center>  

## VAE的数学描述
1. 目标
<img width="1310" height="388" alt="image" src="https://github.com/user-attachments/assets/5aea31bd-e523-4579-a4b4-19cfd619ead4" />  
2. 变分推断
   <img width="1101" height="456" alt="image" src="https://github.com/user-attachments/assets/fc3b7272-2c56-48e7-b095-aa9cb9d8e200" />  
3. 损失函数
   VAE的损失函数由两部分组成：  
   <img width="1296" height="508" alt="image" src="https://github.com/user-attachments/assets/9cc837e7-266e-4bb8-9bd8-4c340f538018" />  
4. 重参数化技巧  
为了使损失函数可通过梯度下降优化，VAE使用重参数化技巧：  
<img width="945" height="204" alt="image" src="https://github.com/user-attachments/assets/a723a5bf-3191-4542-897e-a5c272d14e60" />  
  
5. 模型结构  
   <img width="877" height="219" alt="image" src="https://github.com/user-attachments/assets/5a4a0a7a-82f9-4aca-88c8-2adf281e8e6b" />  
  
6. 生成过程    
生成新样本时：
<img width="504" height="144" alt="image" src="https://github.com/user-attachments/assets/2ddfe035-7641-405d-b64c-f4df6a9cca57" />
--
总结  
VAE通过变分推断优化ELBO，结合编码器和解码器学习数据的潜在表示。损失函数平衡重构质量和潜在分布的正则化，重参数化技巧确保可微性。其数学核心是：
<img width="730" height="73" alt="image" src="https://github.com/user-attachments/assets/fea30f8d-3a62-414b-9f7c-bd8a09805bbb" />  
--- 
## 代码 （Pytorch）

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

# 设置随机种子
torch.manual_seed(42)

# 超参数
input_dim = 784  # 28x28 MNIST图像
hidden_dim = 400
latent_dim = 20
batch_size = 128
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存可视化结果的目录
if not os.path.exists('results'):
    os.makedirs('results')

# VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差
        
        # 解码器
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    # 反归一化目标值从 [-1, 1] 到 [0, 1]
    x = (x.view(-1, input_dim) + 1) / 2  # 反归一化
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 可视化函数
def visualize_results(model, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        # 获取测试集中的一批数据
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        
        # 反归一化用于显示
        data = (data + 1) / 2
        recon_batch = (recon_batch + 1) / 2
        
        # 比较原始图像和重构图像
        comparison = torch.cat([data[:8], recon_batch.view(batch_size, 1, 28, 28)[:8]])
        vutils.save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)
        
        # 生成新的样本
        sample = torch.randn(64, latent_dim).to(device)
        sample = model.decode(sample).cpu()
        sample = (sample + 1) / 2
        vutils.save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png', nrow=8)

# 初始化模型和优化器
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证循环
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')
    return avg_train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    return avg_test_loss

# 主训练循环
if __name__ == "__main__":
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 可视化结果
        visualize_results(model, test_loader, epoch, device)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/loss_curve_{epoch}.png')
        plt.close()
```

### 训练结果

  <img width="934" height="478" alt="image" src="https://github.com/user-attachments/assets/ecda78c7-6330-4a20-9a9c-e62bdbd035d7" />    

图一 训练和验证损失曲线    
<img width="286" height="109" alt="image" src="https://github.com/user-attachments/assets/344ba28c-ea33-492c-afd4-7b352fecc93e" />    

图2 原始图像（上）和生成图像（下）的比较   
