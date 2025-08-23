好的，我已将中文部分翻译成英文，其余保持不变：

---

## Mathematical Description of Convolutional Neural Networks (CNN)

The core of CNNs consists of the following basic operations: **Convolutional Layer**, **Activation Function**, **Pooling Layer**, and finally the **Fully Connected Layer**. Let’s describe them one by one.

---

### 1. Convolutional Layer

Input feature map:

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}
$$

where \$H\$ is the height, \$W\$ is the width, and \$C\_{in}\$ is the number of input channels.

Convolution kernel (filter):

$$
\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
$$

where \$k\_h, k\_w\$ are the kernel sizes, and \$C\_{out}\$ is the number of output channels.

The convolution operation is defined as:

$$
Y_{i,j,c_{out}} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c_{in}=0}^{C_{in}-1} 
X_{i+m, j+n, c_{in}} \cdot K_{m,n,c_{in},c_{out}} + b_{c_{out}}
$$

where \$b\_{c\_{out}}\$ is the bias term. The output feature map is:

$$
\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}
$$

The exact size depends on stride and padding.

---

### 2. Activation Function

A commonly used activation function is ReLU (Rectified Linear Unit):

$$
f(z) = \max(0, z)
$$

Applied to the convolution output:

$$
Z_{i,j,c} = f(Y_{i,j,c})
$$

---

### 3. Pooling Layer

Pooling is used to reduce the feature map size.
For example, max pooling:

$$
P_{i,j,c} = \max_{0 \leq m < p_h, \; 0 \leq n < p_w} Z_{i \cdot s + m, \; j \cdot s + n, \; c}
$$

where \$p\_h, p\_w\$ are the pooling window sizes, and \$s\$ is the stride.

---

### 4. Fully Connected Layer

After several convolution and pooling layers, we obtain a flattened feature vector:

$$
\mathbf{x} \in \mathbb{R}^d
$$

The fully connected layer output is:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

where \$W \in \mathbb{R}^{k \times d}\$, \$\mathbf{b} \in \mathbb{R}^k\$.

---

### 5. Classification Layer (Softmax)

For classification tasks, the final output is passed through Softmax to produce a probability distribution:

![Softmax Formula](https://latex.codecogs.com/png.latex?\hat{y}_i%20=%20\frac{\exp\(y_i\)}{\sum_{j=1}^{k}%20\exp\(y_j\)})

---

要不要我帮你把这些公式和说明整理成 **LaTeX 文档（.tex 文件）**，方便直接编译成 PDF？


