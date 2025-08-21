好的，我已将文本中的英文翻译成中文，并保留了公式和 LaTeX 格式：

---

自编码器的数学描述

### 基本结构

自编码器由编码器和解码器组成：

* **编码器**: 将输入 \$x \in \mathbb{R}^d\$ 映射到低维潜在表示 \$z \in \mathbb{R}^m\$ （通常 \$m < d\$）。
* **解码器**: 将 \$z\$ 重构为输出 \$\hat{x} \in \mathbb{R}^d\$，目标是 \$\hat{x} \approx x\$。

### 2. 数学表达式

* **编码**: \$z = f(x)\$
* **解码**: \$\hat{x} = g(z) = g(f(x))\$
* **损失函数**: 最小化重构误差，通常为均方误差 (MSE)：

$$
\mathcal{L}(x,\hat{x})=\lVert x-\hat{x}\rVert_2^2
=\frac{1}{n}\sum_{i=1}^{n}\bigl(x_i-\hat{x}_i\bigr)^2
$$

其中，\$n\$ 是样本数，\$x\_i\$ 和 \$\hat{x}\_i\$ 分别是输入和重构输出的第 \$i\$ 个元素。

---

### 3. 参数化

* **编码器**: \$f(x) = \sigma(W\_e x + b\_e)\$

  * \$W\_e \in \mathbb{R}^{m \times d},\ b\_e \in \mathbb{R}^m\$，\$\sigma\$ 是激活函数（如 ReLU、Sigmoid）。

* **解码器**: \$g(z) = \sigma'(W\_d z + b\_d)\$

  * \$W\_d \in \mathbb{R}^{d \times m}, b\_d \in \mathbb{R}^d\$，\$\sigma'\$ 是激活函数。

* **优化**: 通过梯度下降调整参数 \$\theta = {W\_e, b\_e, W\_d, b\_d}\$ 来最小化 \$\mathcal{L}\$。

---

### 4. 正则化变体

* 稀疏自编码器：增加稀疏性惩罚以鼓励 \$z\$ 中更少的神经元激活。

* 损失函数：

  $\mathcal{L}_{\text{sparse}} = \mathcal{L}(x, \hat{x}) + \lambda \sum_j \text{KL}(\rho \parallel \hat{\rho}_j)$

  * KL 为 Kullback–Leibler 散度。
  * \$\rho\$ 是目标稀疏度。
  * \$\hat{\rho}\_j\$ 是第 \$j\$ 个神经元的平均激活值。
  * \$\lambda\$ 是正则化系数。

- **去噪自编码器**: 在输入中添加噪声 \$\tilde{x} = x + \epsilon\$ （例如 $\epsilon \sim \mathcal{N}(0, \sigma^2)$），并优化：

  $\mathcal{L}(x, g(f(\tilde{x})))$

---

### 5. 优化

通过反向传播进行优化：

$$
\theta^{*} = \arg \min_{\theta} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\bigl(x_i, g(f(x_i))\bigr)
$$

---

### 6. 应用

* **降维**: \$z\$ 用于特征提取或数据压缩。
* **去噪**: 从 \$\tilde{x}\$ 恢复 \$x\$。
* **异常检测**: 重构误差大的样本可能为异常点。

---

要不要我帮你把这份翻译好的内容整理成一个 **Markdown 版 README**，直接在 GitHub 上排版美观？
