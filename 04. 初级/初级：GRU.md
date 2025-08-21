## 初级：GRU 门控循环单元
<img width="552" height="277" alt="image" src="https://github.com/user-attachments/assets/2c7f3eef-f4be-471c-b7df-cd62b479df28" />

<img width="884" height="266" alt="image" src="https://github.com/user-attachments/assets/75186129-08a6-478c-b91e-82a65e0a601f" />  

门控循环单元（Gated Recurrent Unit, GRU）是一种常用于处理序列数据的循环神经网络（RNN）变体，由Kyunghyun Cho等人于2014年提出。GRU旨在解决传统RNN在长序列处理中遇到的梯度消失或爆炸问题，同时简化了长短期记忆网络（LSTM）的结构，具有更低的计算复杂度和更少的参数。  
### GRU的核心思想
GRU通过引入更新门（update gate）和重置门（reset gate）来控制信息的流动和遗忘，从而有效捕捉序列中的长期依赖关系。与LSTM相比，GRU将遗忘门和输入门合并为单一的更新门，简化了结构，但仍然保留了强大的建模能力。  
### GRU的工作机制
<img width="801" height="874" alt="image" src="https://github.com/user-attachments/assets/59d92294-65c9-4a36-9d51-91e59337ea1d" />
<img width="1229" height="866" alt="image" src="https://github.com/user-attachments/assets/0c326801-56ce-4125-adec-ca7403e67332" />
### GRU的特点
- 简化结构：相比LSTM，GRU只有两个门（更新门和重置门），参数更少，计算效率更高。
- 长期依赖：通过门控机制，GRU能有效捕捉长序列中的依赖关系，缓解梯度消失问题。
- 灵活性：GRU适用于多种序列建模任务，如自然语言处理（NLP）、时间序列预测等。

### GRU与LSTM的对比
-相似点：两者都通过门控机制解决RNN的梯度问题，适合长序列任务。
- 不同点：
GRU结构更简单，参数更少，训练速度更快。  
LSTM有独立的记忆单元，适合更复杂的任务，但计算成本较高。  
在实际应用中，GRU和LSTM性能因任务而异，需根据具体场景选择。  
### 应用场景
- GRU广泛应用于：
自然语言处理：如机器翻译、文本生成、情感分析。  
时间序列分析：如股票预测、天气预测。  
语音处理：如语音识别、语音合成。  

### 总结
GRU是一种高效、简化的循环神经网络变体，通过更新门和重置门实现信息的选择性传递和遗忘。它在保持强大序列建模能力的同时，降低了计算复杂度，是许多序列任务的理想选择。  
## 实例
提供一个使用Python、PyTorch和真实数据集（正弦波序列）的简单GRU示例，展示其原理，并通过Matplotlib可视化预测结果。示例使用正弦波数据进行序列预测，GRU学习序列模式并预测后续值。代码包括数据准备、GRU模型定义、训练和可视化。  

### 说明

数据集：使用正弦波（sin(t)）作为真实数据，生成1000个点。每个样本包含10个连续点作为输入，预测下一个点。  
- GRU模型：  
输入尺寸为1（单变量时间序列）。  
隐藏层尺寸为16（简单网络，足以捕捉正弦波模式）。  
输出尺寸为1（预测下一个值）。 
GRU层处理序列，线性层（fc）将最后一个时间步的输出映射到预测值。  

- 训练：使用Adam优化器和均方误差（MSE）损失函数，训练100个epoch。
可视化：通过Matplotlib绘制真实值（蓝色实线）和预测值（红色虚线），展示GRU对正弦波模式的拟合效果。
## 代码
```
# 修复OpenMP错误 - 必须在所有其他导入之前
import os
import sys

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_WARNINGS'] = 'off'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 设置matplotlib后端和字体
plt.switch_backend('Agg')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

def generate_sine_wave_data():
    """生成正弦波数据"""
    t = np.linspace(0, 20, 1000)  # 时间轴
    data = np.sin(t)  # 正弦波
    sequence_length = 10  # 序列长度
    X, y = [], []

    # 准备输入-输出对：用前10个点预测下一个点
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X).reshape(-1, sequence_length, 1)  # [样本数, 序列长度, 特征数]
    y = np.array(y).reshape(-1, 1)  # [样本数, 1]
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    return X, y, t, data

class GRUModel(nn.Module):
    """GRU模型用于时间序列预测"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU前向传播
        out, _ = self.gru(x, h0)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, X, y, epochs=100, lr=0.01):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    
    print("开始训练GRU模型...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        output = model(X)
        loss = criterion(output, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses

def evaluate_model(model, X, y):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
        true_values = y.numpy()
        
        # 计算评估指标
        mse = np.mean((predictions - true_values) ** 2)
        mae = np.mean(np.abs(predictions - true_values))
        rmse = np.sqrt(mse)
        
        print(f"模型评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return predictions, mse, mae, rmse

def visualize_results(t, data, y, predictions, sequence_length, train_losses):
    """可视化结果"""
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 预测结果对比
    axes[0, 0].plot(t[sequence_length:], y.numpy(), label='真实值', color='blue', linewidth=2)
    axes[0, 0].plot(t[sequence_length:], predictions, label='预测值', color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('GRU正弦波预测结果', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('数值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 训练损失曲线
    axes[0, 1].plot(train_losses, 'b-', linewidth=2)
    axes[0, 1].set_title('训练损失曲线', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. 预测误差
    error = y.numpy() - predictions
    axes[1, 0].plot(t[sequence_length:], error, color='green', linewidth=1)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_title('预测误差', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('时间')
    axes[1, 0].set_ylabel('误差')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 误差分布直方图
    axes[1, 1].hist(error, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('误差分布直方图', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('误差')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_sine_wave_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_architecture():
    """可视化模型架构"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 定义颜色
    colors = {
        'input': '#E8F4FD',
        'gru': '#FFE6E6',
        'fc': '#E6FFE6',
        'output': '#F0E6FF'
    }
    
    # 定义位置
    positions = {
        'input': (2, 6),
        'gru1': (4, 6),
        'gru2': (6, 6),
        'fc': (8, 6),
        'output': (10, 6)
    }
    
    # 绘制网络层
    for name, pos in positions.items():
        if 'gru' in name:
            color = colors['gru']
            width, height = 2.0, 1.2
        elif 'fc' in name:
            color = colors['fc']
            width, height = 2.0, 1.2
        elif 'input' in name:
            color = colors['input']
            width, height = 1.8, 1.0
        elif 'output' in name:
            color = colors['output']
            width, height = 1.8, 1.0
        else:
            color = colors['input']
            width, height = 1.8, 1.0
        
        # 绘制框
        box = FancyBboxPatch(
            (pos[0] - width/2, pos[1] - height/2),
            width, height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # 添加文本
        if 'gru' in name:
            text = f'{name.upper()}\nHidden Size: 32'
        elif 'fc' in name:
            text = f'{name.upper()}\n32→1'
        elif 'input' in name:
            text = 'Input\n10×1'
        elif 'output' in name:
            text = 'Output\n1'
        else:
            text = name
        
        ax.text(pos[0], pos[1], text, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # 绘制连接线
    main_flow = ['input', 'gru1', 'gru2', 'fc', 'output']
    for i in range(len(main_flow) - 1):
        start_pos = positions[main_flow[i]]
        end_pos = positions[main_flow[i + 1]]
        ax.arrow(start_pos[0] + 0.9, start_pos[1], 
                end_pos[0] - start_pos[0] - 1.8, 0,
                head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=3)
    
    # 设置坐标轴
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('GRU模型架构', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('gru_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始GRU正弦波预测实验...")
    
    # 1. 生成数据
    print("生成正弦波数据...")
    X, y, t, data = generate_sine_wave_data()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 创建模型
    print("创建GRU模型...")
    model = GRUModel(input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.1)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 生成模型架构图
    print("生成模型架构图...")
    visualize_model_architecture()
    
    # 4. 训练模型
    train_losses = train_model(model, X, y, epochs=100, lr=0.01)
    
    # 5. 评估模型
    predictions, mse, mae, rmse = evaluate_model(model, X, y)
    
    # 6. 可视化结果
    print("生成可视化结果...")
    visualize_results(t, data, y, predictions, 10, train_losses)
    
    # 7. 保存模型
    torch.save(model.state_dict(), 'gru_sine_wave_model.pth')
    print("模型已保存到: gru_sine_wave_model.pth")
    
    print("GRU正弦波预测实验完成！")
    print("生成的文件:")
    print("- gru_model_architecture.png: 模型架构图")
    print("- gru_sine_wave_results.png: 预测结果可视化")
    print("- gru_sine_wave_model.pth: 训练好的模型")
    
    # 清理内存
    del model, X, y
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main() 
```
### 训练结果
GRU通过更新门和重置门学习正弦波的周期性模式。  

<img width="1124" height="404" alt="image" src="https://github.com/user-attachments/assets/7edf7421-1de7-49db-a6ad-59f663023739" />  

<img width="1941" height="1283" alt="image" src="https://github.com/user-attachments/assets/539fc09d-fab6-4b79-ab8a-5d50d92fe6ec" />



