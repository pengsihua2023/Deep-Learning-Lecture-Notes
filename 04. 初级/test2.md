---

# 卷积神经网络（CNN）的数学描述

卷积神经网络（Convolutional Neural Network, **CNN**）是一类常用于处理图像、时序等具有局部相关性的深度学习模型。它的数学描述可以从 **卷积层、激活函数、池化层、全连接层** 逐步构建。

---

## 1. 卷积运算（Convolution）
对于输入张量 $X \in \mathbb{R}^{H \times W \times C_{in}}$（高度 $H$、宽度 $W$、通道数 $C_{in}$），卷积核（滤波器） $K \in \mathbb{R}^{k_h \times k_w \times C_{in}}$，输出特征图的某一通道 $Y_{ij}$ 定义为：

$$
Y_{ij} = \sum_{m=1}^{k_h}\sum_{n=1}^{k_w}\sum_{c=1}^{C_{in}} K_{mnc}\, X_{(i+m-1)(j+n-1)c} + b
$$

其中：
- $k_h, k_w$：卷积核的高和宽。  
- $b$：偏置项。  
- $i,j$：输出特征图的位置索引。  

若有多个卷积核（数量为 $C_{out}$），则输出特征图维度为  
$$
Y \in \mathbb{R}^{H' \times W' \times C_{out}}
$$  
其中 $H', W'$ 由步幅（stride, $s$）和填充（padding, $p$）决定：  
$$
H' = \frac{H - k_h + 2p}{s} + 1,\quad W' = \frac{W - k_w + 2p}{s} + 1
$$

---

## 2. 激活函数（Activation）
卷积层输出经过非线性变换：
$$
Z_{ij}^{(c)} = \sigma(Y_{ij}^{(c)})
$$

常见的 $\sigma(\cdot)$：  
- ReLU: $\sigma(x) = \max(0, x)$  
- Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$  
- Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$  

---

## 3. 池化层（Pooling）
用于下采样，减少计算量并提取主要特征。  

常见最大池化（Max Pooling）：  
$$
P_{ij} = \max_{(m,n) \in \Omega_{ij}} Z_{mn}
$$

其中 $\Omega_{ij}$ 表示以 $(i,j)$ 为中心的池化窗口区域。  

平均池化（Average Pooling）：  
$$
P_{ij} = \frac{1}{|\Omega_{ij}|} \sum_{(m,n)\in\Omega_{ij}} Z_{mn}
$$

---

## 4. 全连接层（Fully Connected, FC）
将特征图展平成向量 $x \in \mathbb{R}^d$，经过仿射变换：
$$
h = W x + b
$$

其中 $W \in \mathbb{R}^{k \times d}, b \in \mathbb{R}^k$。  

输出层通常接 softmax（用于分类问题）：  
$$
\hat{y}_i = \frac{e^{h_i}}{\sum_{j=1}^k e^{h_j}}
$$

---

## 5. CNN 整体结构
一个典型 CNN 的前向传播可表述为：  
$$
X \xrightarrow{\text{Conv+ReLU}} Z_1 \xrightarrow{\text{Pooling}} P_1 
\xrightarrow{\text{Conv+ReLU}} Z_2 \xrightarrow{\text{Pooling}} P_2 
\xrightarrow{\text{Flatten}} x \xrightarrow{\text{FC+Softmax}} \hat{y}
$$

---

✅ 总结：  
CNN 的数学描述核心是 **卷积（局部加权求和）、激活（非线性变换）、池化（降采样）、全连接层（全局映射）**，最终形成端到端的分类或回归模型。
