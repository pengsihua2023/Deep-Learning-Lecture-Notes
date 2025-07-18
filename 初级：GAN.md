## GAN
生成对抗网络（Generative Adversarial Network, GAN）  
- 重要性：GAN 是生成模型的代表，展示深度学习的创造力，适合激发中学生对 AI 的兴趣。  
- 核心概念：
GAN 有两个网络：生成器（造假图片）和判别器（判断真假），两者“竞争”学习。   
生成器最终能生成逼真的数据（如图片、音乐）。    
- 应用：生成艺术作品、修复老照片、游戏角色设计。  
 <img width="1213" height="529" alt="image" src="https://github.com/user-attachments/assets/8cca8f1f-73ad-4df2-9b66-55353dd7b7c8" />

## 代码(Pytorch)
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 1. 参数设置
z_dim = 100  # 噪声输入维度
image_dim = 28 * 28  # MNIST图像大小 (28x28)
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5  # Adam优化器的beta1参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. 定义生成器
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.ReLU(True),
            nn.Linear(1024, image_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

# 4. 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 5. 初始化模型、损失函数和优化器
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 6. 训练GAN
def train_gan():
    real_label = 1.0
    fake_label = 0.0
    
    for epoch in range(num_epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # 训练判别器
            discriminator.zero_grad()
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            d_real_output = discriminator(real_images)
            d_real_loss = criterion(d_real_output, real_labels)
            
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)
            d_fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            generator.zero_grad()
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)  # 希望生成器生成的被判别为真
            g_loss.backward()
            g_optimizer.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            num_batches += 1
        
        # 每5个epoch打印损失
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'D Loss: {d_loss_total/num_batches:.4f}, '
                  f'G Loss: {g_loss_total/num_batches:.4f}')
        
        # 每10个epoch保存生成的图像
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(torch.randn(16, z_dim).to(device)).cpu()
                save_images(fake_images, epoch + 1)

# 7. 保存生成的图像
def save_images(images, epoch):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    images = images * 0.5 + 0.5  # 反归一化到[0, 1]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close()

# 8. 可视化结果
def plot_final_results():
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_images = generator(z).cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fake_images = fake_images * 0.5 + 0.5  # 反归一化到[0, 1]
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.title('Generated MNIST Digits')
    plt.show()

# 9. 执行训练和可视化
if __name__ == "__main__":
    print("Training started...")
    train_gan()
    print("\nGenerating final visualization...")
    plot_final_results()
```
## 训练结果
Epoch [45/50], D Loss: 0.3414, G Loss: 2.7712   
Epoch [50/50], D Loss: 0.3680, G Loss: 2.7854   

Generating final visualization...   
