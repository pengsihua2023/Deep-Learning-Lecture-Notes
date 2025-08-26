## Diffusion Models (扩散模型)
- 提出者：多位研究者（如DDPM）  
   
理论最早提出者：Jascha Sohl-Dickstein  
<div align="center">
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/06fe6aa7-bb8f-45de-9239-c490de348e6e" />
 
DDPM 第一作者：Jonathan Ho  
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/ee589462-59e3-432c-b20d-87b11a0ad85d" />  
</div>

扩散模型（Diffusion Models）是一种强大的生成式人工智能模型，主要用于从噪声中逐步生成高质量数据，如图像、音频或文本。它们的核心机制是通过模拟一个“扩散过程”来实现数据生成：先将真实数据逐步添加噪声（前向过程），然后训练模型学习逆向过程（去噪），从而从纯噪声中恢复出类似训练数据的样本。这种方法在生成模型领域取得了突破，尤其在图像合成任务中表现突出。扩散模型的理论基础最早由 Jascha Sohl-Dickstein 等人在 2015 年的论文《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》中提出，因此 Jascha Sohl-Dickstein 被视为扩散模型的原创第一作者。该论文引入了基于非平衡热力学的扩散概念，用于无监督学习。 然而，扩散模型的流行始于 2020 年 Jonathan Ho 等人的《Denoising Diffusion Probabilistic Models》（DDPM），这篇论文将扩散模型应用于实际图像生成，并显著提升了生成质量。  
   

- 特点：通过加噪和去噪过程生成高质量图像，性能超越GAN。  
- 掌握要点：去噪过程、概率建模。  
- 重要性：
扩散模型是生成高质量图像的最新技术，驱动了 DALL·E 2、Stable Diffusion 等生成 AI。  
在图像生成、文本到图像等领域超越了 GAN，成为生成模型的新标杆。  
- 核心概念：
扩散模型通过“加噪-去噪”过程学习数据分布，先把数据加噪到随机噪声，再逐步还原。  
- 应用：图像生成（艺术、游戏设计）、视频生成、科学模拟。  
<img width="700" height="320" alt="image" src="https://github.com/user-attachments/assets/427d35b9-10d1-4bca-b74c-b5e166d7613d" />

## 代码
该代码实现了一个简单的**扩散模型（DDPM，Denoising Diffusion Probabilistic Model）**，使用PyTorch库，主要功能如下：  

1. **数据生成**：生成2D正态分布数据（均值[2, 2]，标准差0.5）作为训练和可视化样本。
2. **前向扩散**：通过逐渐添加高斯噪声（1000步，线性方差调度），将原始数据逐步转变为纯噪声。
3. **去噪模型**：使用一个简单的多层感知机（MLP）预测每一步的噪声，输入为带时间步信息的噪声数据，输出为预测的噪声。
4. **训练**：通过最小化预测噪声与真实噪声的均方误差，训练模型（1000个epoch，Adam优化器）。
5. **采样**：从纯噪声开始，通过逆向去噪过程生成与原始数据分布相似的样本。
6. **可视化**：使用Matplotlib绘制原始数据和生成样本的2D散点图，比较它们的分布。

代码支持CPU/GPU运行，确保设备一致性，并输出训练损失和生成样本的形状。最终散点图直观展示模型是否成功学习到数据分布。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 超参数
num_steps = 1000  # 扩散步数
beta_start = 0.0001  # 方差调度起始值
beta_end = 0.02  # 方差调度终止值
data_dim = 2  # 数据维度（2D正态分布）
batch_size = 128
epochs = 1000
lr = 0.001
n_samples = 1000  # 用于可视化的样本数

# 线性方差调度
betas = torch.linspace(beta_start, beta_end, num_steps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 前向扩散过程
def forward_diffusion(x_0, t, device):
    betas_t = betas.to(device)
    alphas_cumprod_t = alphas_cumprod.to(device)
    noise = torch.randn_like(x_0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_t[t]).view(-1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod_t[t]).view(-1, 1)
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise, noise

# 简单MLP模型
class SimpleDenoiser(nn.Module):
    def __init__(self):
        super(SimpleDenoiser, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim)
        )
    
    def forward(self, x, t):
        t = t.view(-1, 1).float() / num_steps
        x_t = torch.cat([x, t], dim=1)
        return self.model(x_t)

# 生成简单数据集（2D正态分布）
def generate_data(n_samples):
    return torch.randn(n_samples, data_dim) * 0.5 + torch.tensor([2.0, 2.0])

# 训练模型
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move global tensors to device
    global betas, alphas, alphas_cumprod
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    model = SimpleDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        data = generate_data(batch_size).to(device)
        t = torch.randint(0, num_steps, (batch_size,), device=device)
        
        x_t, noise = forward_diffusion(data, t, device)
        predicted_noise = model(x_t, t)
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    return model

# 采样过程
def sample(model, n_samples, device):
    betas_t = betas.to(device)
    alphas_t = alphas.to(device)
    alphas_cumprod_t = alphas_cumprod.to(device)
    
    x = torch.randn(n_samples, data_dim).to(device)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)
        predicted_noise = model(x, t_tensor)
        alpha = alphas_t[t]
        alpha_cumprod = alphas_cumprod_t[t]
        beta = betas_t[t]
        
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
        if t > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)
    return x

# 可视化函数
def visualize_samples(original_data, generated_samples):
    plt.figure(figsize=(10, 5))
    
    # 原始数据散点图
    plt.subplot(1, 2, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, label="Original Data")
    plt.title("Original Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    # 生成样本散点图
    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, color="orange", label="Generated Samples")
    plt.title("Generated Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练模型
    model = train()
    
    # 生成用于可视化的原始数据和生成样本
    original_data = generate_data(n_samples).cpu().numpy()
    generated_samples = sample(model, n_samples, device).cpu().detach().numpy()
    
    # 可视化结果
    visualize_samples(original_data, generated_samples)
    print("Generated samples shape:", generated_samples.shape)

```

## 训练结果

Epoch 700, Loss: 0.1903  
Epoch 800, Loss: 0.1887  
Epoch 900, Loss: 0.2179  
Epoch 1000, Loss: 0.3247  
Generated samples shape: (1000, 2)  

<img width="984" height="493" alt="image" src="https://github.com/user-attachments/assets/b57caf73-74d1-41a4-b547-ff59cd9670a8" />

图2 原始样本和生成的比较
