## Autoencoder 
Autoencoder（自编码器）
- 重要性：
Autoencoder 是一种无监督学习模型，用于数据压缩、降噪或特征学习。  
它是生成模型（如 GAN）的先驱，广泛用于数据预处理和异常检测。  
- 核心概念：
Autoencoder 包含编码器（压缩数据）和解码器（重构数据），目标是让输出尽可能接近输入。  
比喻：像一个“数据压缩机”，把大文件压缩后再解压，尽量保持原样。  
- 应用：图像去噪、数据压缩、异常检测（如信用卡欺诈检测）。
<img width="1400" height="797" alt="image" src="https://github.com/user-attachments/assets/28b89fa6-5c8b-460f-8385-4cd46c7c47cd" />

图1 第一种表示 
<img width="700" height="220" alt="image" src="https://github.com/user-attachments/assets/f20e1904-4878-4950-a91f-cbe0d2336f50" />

图2 第二种表示

<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/dbd389b4-042e-44bf-a62f-ef736bbebd89" />

图3 第三种表示
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

# Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 损失函数（仅使用重构损失）
def loss_function(recon_x, x):
    # 反归一化目标值从 [-1, 1] 到 [0, 1]
    x = (x.view(-1, input_dim) + 1) / 2
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

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
        recon_batch = model(data)
        
        # 反归一化用于显示
        data = (data + 1) / 2
        recon_batch = (recon_batch + 1) / 2
        
        # 比较原始图像和重构图像
        comparison = torch.cat([data[:8], recon_batch.view(batch_size, 1, 28, 28)[:8]])
        vutils.save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)

# 初始化模型和优化器
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证循环
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
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
            recon_batch = model(data)
            test_loss += loss_function(recon_batch, data).item()
    
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
## 运行结果
====> Epoch: 9 Average training loss: 69.1569  
====> Test set loss: 69.0569 
Train Epoch: 10 [0/60000 (0%)]  Loss: 71.628830  
Train Epoch: 10 [12800/60000 (21%)]     Loss: 65.910645  
Train Epoch: 10 [25600/60000 (43%)]     Loss: 68.564079  
Train Epoch: 10 [38400/60000 (64%)]     Loss: 70.579895  
Train Epoch: 10 [51200/60000 (85%)]     Loss: 69.532722  
====> Epoch: 10 Average training loss: 68.6832  
====> Test set loss: 68.4474  

<img width="960" height="490" alt="image" src="https://github.com/user-attachments/assets/8d28cf45-b977-4de8-a857-d62f8893be0f" />    

图4 loss曲线  
<img width="274" height="108" alt="image" src="https://github.com/user-attachments/assets/d5769c88-f37c-4d0a-94b9-fb627129abfd" />  


图5 输入与输出图像比较


