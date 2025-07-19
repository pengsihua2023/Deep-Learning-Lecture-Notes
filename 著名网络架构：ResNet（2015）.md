## ResNet （残差网络，Residual Network）
- 提出者：何凯明（微软研究院）  
- 特点：引入残差连接（Residual Connections），解决深层网络梯度消失问题，可构建数百层网络。 
- 掌握要点：残差学习、深层网络训练技巧。  
- 重要性：
ResNet 是 CNN 的进阶版本，通过“残差连接”解决深层网络的梯度消失问题，允许构建非常深的网络（几十到上百层）。  
在图像分类（如 ImageNet 比赛）中表现卓越，是现代计算机视觉的基石。  
- 核心概念：
ResNet 引入“残差连接”（skip connection），让网络学习“变化量”而非直接输出，减轻深层网络的训练难度。  
- 应用：图像分类、目标检测（如自动驾驶中的物体识别）、人脸识别。
<img width="570" height="328" alt="image" src="https://github.com/user-attachments/assets/4c111489-898f-4412-9e70-336ec2320f03" />  
 <img width="850" height="595" alt="image" src="https://github.com/user-attachments/assets/ee800edc-db6e-4cde-84d9-0d396ca69e58" />  


## 代码

该代码实现了一个**简化的ResNet模型**，用于在**CIFAR-10数据集**上进行图像分类任务。主要功能如下：

1. **残差块定义**：
   - 实现`residual_block`函数，定义一个残差模块，包含两个卷积层（带批量归一化和ReLU激活）及残差连接。
   - 支持维度调整（通过1x1卷积调整shortcut分支），确保输入输出维度匹配。

2. **模型构建**：
   - 定义`build_simple_resnet`函数，构建一个简化的ResNet模型：
     - 初始卷积层（64个滤波器，3x3卷积）。
     - 堆叠4个残差块（两组64通道和两组128通道，第二组使用stride=2降采样）。
     - 全局平均池化后接全连接层，输出10类分类结果（带softmax激活）。

3. **数据预处理**：
   - 加载CIFAR-10数据集（32x32彩色图像）。
   - 将像素值归一化到[0,1]范围。

4. **模型编译与训练**：
   - 使用Adam优化器和稀疏分类交叉熵损失函数编译模型，跟踪准确率指标。
   - 以batch_size=64训练模型10个epoch，使用测试集进行验证。

代码基于TensorFlow/Keras实现，适用于CIFAR-10图像分类，输出模型结构摘要并进行训练，旨在学习图像的分类特征。

 
```
import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters, kernel_size=3, stride=1):
    """定义一个简单的残差块"""
    shortcut = x
    
    # 第一个卷积层
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 第二个卷积层
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 如果维度不匹配，调整shortcut的维度
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # 残差连接
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_simple_resnet(input_shape=(32, 32, 3), num_classes=10):
    """构建简单的ResNet模型"""
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 堆叠残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    
    # 全连接层
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x)
    return model

# 示例：创建并编译模型
if __name__ == "__main__":
    model = build_simple_resnet()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()
    
    # 假设使用CIFAR-10数据集进行测试
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # 数据预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 训练模型
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
```
