好的，我来写出卷积神经网络（CNN）的数学描述。

---

# 卷积神经网络（CNN）的数学描述

CNN 的核心由以下几个基本运算组成：**卷积层（Convolutional Layer）**、**非线性激活函数（Activation Function）**、**池化层（Pooling Layer）**，以及最后的 **全连接层（Fully Connected Layer）**。我们逐一描述。

---

## 1. 卷积层（Convolutional Layer）

设输入特征图为

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}
$$

其中 $H$ 是高度，$W$ 是宽度，$C_{in}$ 是输入通道数。

卷积核（滤波器）为

$$
\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
$$

其中 $k_h, k_w$ 为卷积核大小，$C_{out}$ 为输出通道数。

卷积运算定义为：

$$
Y_{i,j,c_{out}} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c_{in}=0}^{C_{in}-1} 
X_{i+m, j+n, c_{in}} \cdot K_{m,n,c_{in},c_{out}} + b_{c_{out}}
$$

其中 $b_{c_{out}}$ 是偏置项。输出特征图为

$$
\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}
$$

具体尺寸取决于步幅（stride）和填充（padding）。

---

## 2. 激活函数（Activation Function）

常用激活函数为 ReLU（线性整流单元）：

$$
f(z) = \max(0, z)
$$

应用到卷积输出：

$$
Z_{i,j,c} = f(Y_{i,j,c})
$$

---

## 3. 池化层（Pooling Layer）

池化操作用于降低特征图尺寸。
以最大池化（Max Pooling）为例：

$$
P_{i,j,c} = \max_{0 \leq m < p_h, \; 0 \leq n < p_w} Z_{i \cdot s + m, \; j \cdot s + n, \; c}
$$

其中 $p_h, p_w$ 为池化窗口大小，$s$ 为步幅。

---

## 4. 全连接层（Fully Connected Layer）

经过若干层卷积和池化后，得到展平的特征向量：

$$
\mathbf{x} \in \mathbb{R}^d
$$

全连接层输出为：

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

其中 $W \in \mathbb{R}^{k \times d}$，$\mathbf{b} \in \mathbb{R}^k$。

---

## 5. 分类层（Softmax）

在分类任务中，最后通过 Softmax 输出概率分布：

$$
\hat{y}_i = \frac{\exp(y_i)}{\sum_{j=1}^k \exp(y_j)}
$$

$$
\frac{\exp(y_i)}{\sum_{j=1}^k \exp(y_j)}
$$

$\hat{y}_i = $

$\frac{\exp(y_i)}{\sum_{j=1}^k \exp(y_j)}$

$\hat{y}_i = $  $\frac{\exp(y_i)}{\sum_{j=1}^k \exp(y_j)}$

$\hat{y}_i = \frac{e^{y_i}}{\sum_{j=1}^k e^{y_j}}, \quad i=1,2,\dots,k$

$\hat{y}_i = \frac{e^{y_i}}{\sum_{j=1}^k e^{y_j}}$

---





