{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 变分自编码器（Variational Autoencoder, VAE）实现\n",
    "\n",
    "## 介绍\n",
    "变分自编码器（VAE）是一种生成模型，结合了深度学习和变分推断，能够学习数据的潜在表示并生成新样本。它由编码器（将输入映射到潜在空间分布）、采样过程和解码器（从潜在变量重建数据）组成。VAE的目标是最大化数据的边际似然，通过优化证据下界（ELBO）实现。\n",
    "\n",
    "## 数学描述\n",
    "VAE的目标是最大化边际似然 $p(x)$，通过优化ELBO：\n",
    "\n",
    "$$\n",
    "\\mathcal{L}(\\theta, \\phi; x) = \\mathbb{E}_{q_\\phi(z|x)} [\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\| p(z))\n",
    "$$\n",
    "\n",
    "其中：\n",
    "- $\\theta$ 是解码器参数，$\\phi$ 是编码器参数。\n",
    "- 第一项是重建损失，衡量重建数据的准确性（例如均方误差或二元交叉熵）。\n",
    "- 第二项是KL散度，使后验分布 $q(z|x)$ 接近先验 $p(z)$（通常为标准正态分布 $N(0, I)$）。\n",
    "\n",
    "KL散度公式（高斯分布）：\n",
    "\n",
    "$$\n",
    "D_{KL}(q(z|x) \\| p(z)) = -\\frac{1}{2} \\sum_{j=1}^J (1 + \\log(\\sigma_j^2) - \\mu_j^2 - \\sigma_j^2)\n",
    "$$\n",
    "\n",
    "采样使用重参数化技巧：$z = \\mu + \\sigma \\odot \\epsilon$，其中 $\\epsilon \\sim N(0, I)$。\n",
    "\n",
    "## 实现\n",
    "以下是一个简单的VAE实现，使用PyTorch，针对MNIST数据集（28x28灰度图像）。编码器和解码器为MLP，潜在维度为2。代码包括模型定义、损失函数和训练循环。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import datasets, transforms\n",
    "\n",
    "# 设置超参数\n",
    "input_dim = 28 * 28  # MNIST图像大小\n",
    "hidden_dim = 400\n",
    "latent_dim = 2  # 潜在空间维度\n",
    "batch_size = 128\n",
    "epochs = 10\n",
    "lr = 1e-3\n",
    "\n",
    "# 数据加载\n",
    "transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])\n",
    "train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)\n",
    "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 模型定义\n",
    "定义VAE模型，包括编码器、解码器和重参数化采样。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class VAE(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(VAE, self).__init__()\n",
    "        # 编码器\n",
    "        self.fc1 = nn.Linear(input_dim, hidden_dim)\n",
    "        self.fc_mu = nn.Linear(hidden_dim, latent_dim)\n",
    "        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)\n",
    "        # 解码器\n",
    "        self.fc3 = nn.Linear(latent_dim, hidden_dim)\n",
    "        self.fc4 = nn.Linear(hidden_dim, input_dim)\n",
    "\n",
    "    def encode(self, x):\n",
    "        h = torch.relu(self.fc1(x))\n",
    "        mu = self.fc_mu(h)\n",
    "        logvar = self.fc_logvar(h)\n",
    "        return mu, logvar\n",
    "\n",
    "    def reparameterize(self, mu, logvar):\n",
    "        std = torch.exp(0.5 * logvar)\n",
    "        eps = torch.randn_like(std)\n",
    "        return mu + eps * std\n",
    "\n",
    "    def decode(self, z):\n",
    "        h = torch.relu(self.fc3(z))\n",
    "        return torch.sigmoid(self.fc4(h))  # 输出在[0,1]间，适合MNIST\n",
    "\n",
    "    def forward(self, x):\n",
    "        mu, logvar = self.encode(x)\n",
    "        z = self.reparameterize(mu, logvar)\n",
    "        return self.decode(z), mu, logvar"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 损失函数\n",
    "损失函数包括重建损失（二元交叉熵）和KL散度。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_function(recon_x, x, mu, logvar):\n",
    "    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')  # 重建损失\n",
    "    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL散度\n",
    "    return BCE + KLD"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 训练循环\n",
    "训练VAE模型，优化ELBO损失。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = VAE()\n",
    "optimizer = optim.Adam(model.parameters(), lr=lr)\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    model.train()\n",
    "    train_loss = 0\n",
    "    for batch_idx, (data, _) in enumerate(train_loader):\n",
    "        optimizer.zero_grad()\n",
    "        recon_batch, mu, logvar = model(data)\n",
    "        loss = loss_function(recon_batch, data, mu, logvar)\n",
    "        loss.backward()\n",
    "        train_loss += loss.item()\n",
    "        optimizer.step()\n",
    "    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 生成样本\n",
    "训练完成后，从潜在空间采样生成新图像。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with torch.no_grad():\n",
    "    z = torch.randn(64, latent_dim)  # 随机采样\n",
    "    samples = model.decode(z).view(64, 1, 28, 28)\n",
    "    # 可以用以下代码保存生成图像（需取消注释）\n",
    "    # from torchvision.utils import save_image\n",
    "    # save_image(samples, 'samples.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 说明\n",
    "此实现是最简化的VAE，适合初学者理解。实际应用中，可使用卷积层（Conv2D）替换MLP以提升图像处理性能，或调整潜在维度、隐藏层大小等超参数以优化结果。运行时需确保安装PyTorch和torchvision，并有GPU支持以加速训练。"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
