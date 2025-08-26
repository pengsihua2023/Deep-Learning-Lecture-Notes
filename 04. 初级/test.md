## 变分自编码器（Variational Autoencoder, VAE）
变分自编码器（VAE）是一种生成式深度学习模型，由Kingma和Welling于2013年提出。它是自编码器（Autoencoder）的变体，但引入了变分推断（Variational Inference）的思想，使其能够生成新数据，而不仅仅是压缩和重建输入。VAE的主要目的是学习数据的潜在表示（latent representation），并通过潜在空间的采样来生成类似于训练数据的样本。

### VAE的核心组件包括：
- 编码器（Encoder）：将输入数据x映射到潜在空间的分布参数（通常是高斯分布的均值μ和方差σ²）。
- 采样（Sampling）：从潜在分布中采样潜在变量z，使用重参数化技巧（reparameterization trick）使采样过程可微分。
- 解码器（Decoder）：从潜在变量z重建输出数据x'，目标是使x'尽可能接近x。
- 损失函数：结合重建损失（reconstruction loss，如MSE）和KL散度（Kullback-Leibler divergence），用于正则化潜在分布，使其接近先验分布（通常是标准正态分布）。
VAE的优势在于它能生成连续的潜在空间，支持插值和生成新样本，常用于图像生成、数据增强等领域。相比GAN（生成对抗网络），VAE的训练更稳定，但生成的样本可能更模糊。
