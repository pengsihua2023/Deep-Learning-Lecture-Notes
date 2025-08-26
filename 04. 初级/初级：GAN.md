## GAN
生成对抗网络（Generative Adversarial Network, GAN）  
- 重要性：GAN 是生成模型的代表，展示深度学习的创造力，适合激发中学生对 AI 的兴趣。  
- 核心概念：
GAN 有两个网络：生成器（造假图片）和判别器（判断真假），两者“竞争”学习。   
生成器最终能生成逼真的数据（如图片、音乐）。    
- 应用：生成艺术作品、修复老照片、游戏角色设计。  
 <img width="600" height="260" alt="image" src="https://github.com/user-attachments/assets/8cca8f1f-73ad-4df2-9b66-55353dd7b7c8" />

这张图片展示了生成对抗网络（GAN）这一机器学习模型的概念。它包含两个主要组成部分：  
1. **生成器**：在左侧用青色表示，生成器负责创建数据样本（例如图中的方块）。它的作用是生成模仿真实数据的假数据。  
2. **判别器**：在右侧用红色表示，判别器对接收到的数据进行评估，判断其是“真实”（真实数据）还是“伪造”（生成数据）。绿色标签“真实”和“伪造”表示判别器的分类结果。  
在GAN中，生成器和判别器会同时进行训练，通过竞争过程进行优化：生成器通过试图欺骗判别器来改进，而判别器通过更好地区分真实数据和假数据来提升。这种迭代过程会持续进行，直到生成器能够生成高度逼真的数据。
## GAN的数学描述
1. 基本结构  
GAN由两个模型组成：  
生成器（G）：将随机噪声 $ z $（通常从标准正态分布或均匀分布中采样）映射到数据空间，生成假样本G(z) ，试图模仿真实数据的分布 Pdata。  
判别器（D）：接收输入（真实样本x 或生成样本G(z)），输出一个标量 D(x) 或  D(G(z)) ，表示输入是真实样本的概率（接近1）还是生成样本（接近0）。
2. 优化目标  
GAN的核心是一个最小最大博弈问题，生成器和判别器通过对抗性训练优化。目标函数可以表示为：
<img width="1058" height="342" alt="image" src="https://github.com/user-attachments/assets/2ac71e60-4723-46ac-8602-1fea4bd4368a" />  


$$
\min_G \max_D V(D, G) = \mathbb{E}_ {x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

**解释：**

* $\mathbb{E}_ {x \sim p_{\text{data}}(x)}[\log D(x)]$：判别器试图最大化对真实样本的正确分类概率。

* $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$：判别器试图最大化对生成样本的拒绝概率，而生成器试图让 $D(G(z))$ 接近 1（即欺骗判别器）。

* 生成器 $G$ 希望最小化 $\log(1 - D(G(z)))$，使生成样本尽可能接近真实数据。


3. 目标函数的直观理解  
   <img width="1152" height="191" alt="image" src="https://github.com/user-attachments/assets/73e1b6f2-7dfd-415b-8199-eb5fbad47657" />



* 判别器 $D$ 的目标是区分真实数据 $x \sim p_{\text{data}}$ 和生成数据 $G(z) \sim p_g$，最大化上述目标函数。

* 生成器 $G$ 的目标是让生成分布 $p_g$ 尽可能接近真实数据分布 $p_{\text{data}}$，即欺骗判别器，使 $D(G(z)) \approx 1$。

在理想情况下，当 $p_g = p_{\text{data}}$，判别器无法区分真假，输出 $D(x) = D(G(z)) = 0.5, $
此时达到纳什均衡。
   
4. 训练过程   
GAN的训练通过交替优化以下两步进行：  
<img width="1144" height="524" alt="image" src="https://github.com/user-attachments/assets/11e4dd4d-8a6e-43e3-88a3-1eaa24d4a42d" />



**1. 优化判别器：**

* 固定生成器 $G$，用真实样本 $x \sim p_{\text{data}}$ 和生成样本 $G(z) \sim p_z$ 训练判别器，最大化：

$$
V(D) = \mathbb{E}_ {x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

* 通常使用梯度上升更新 $D$ 的参数。



**2. 优化生成器：**

* 固定判别器 $D$，用噪声 $z \sim p_z$ 生成样本 $G(z)$，最小化：

$$
V(G) = \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

* 实践中，常优化等价形式 $\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$

因为原始形式梯度可能不稳定（尤其当 $D(G(z)) \approx 0$ 时）。





5. 数学性质与挑战  
   <img width="1109" height="424" alt="image" src="https://github.com/user-attachments/assets/eb9fde5f-e3a3-4ff1-a822-5c71343c48a6" />



* **全局最优**：当 $p_g = p_{\text{data}}$，目标函数 $V(D,G)$ 达到全局最优，判别器输出 $D(x) = 0.5$。

* **JS散度**：GAN 的优化可以看作最小化生成分布 $p_g$ 和真实分布 $p_{\text{data}}$ 之间的 Jensen–Shannon 散度：

$$
JS(p_{\text{data}} \parallel p_g) = \frac{1}{2} KL\left(p_{\text{data}} \parallel \frac{p_{\text{data}} + p_g}{2}\right) + \frac{1}{2} KL\left(p_g \parallel \frac{p_{\text{data}} + p_g}{2}\right)
$$


* **挑战**：

  * **模式崩塌**：生成器可能只生成有限的样本模式，忽略真实数据的多样性。
  * **训练不稳定**：由于对抗性目标，梯度可能震荡或消失。
  * **梯度消失**：当判别器过强，生成器可能无法有效学习。




  
6. 总结  
   <img width="1188" height="268" alt="image" src="https://github.com/user-attachments/assets/dc02ea44-ffaa-42a6-95bc-91ef5a349867" />





## 代码(Pytorch)
```python
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
<img width="741" height="708" alt="image" src="https://github.com/user-attachments/assets/1787c031-5bf5-4ef9-b734-ef7e6ef65b3b" />
