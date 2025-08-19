## 多GPU并行训练 （Distributed Data Parallel (DDP)）
### 什么是分布式数据并行（Distributed Data Parallel, DDP）？

**分布式数据并行（DDP）**是 PyTorch 中一种用于分布式训练的技术，通过在多个 GPU 或多台机器上并行处理数据，加速深度学习模型的训练。它是数据并行的一种实现，适合大规模模型和数据集，能够高效利用多 GPU 或分布式环境。

#### 核心特点：
- **适用场景**：适用于需要加速训练的大型深度学习任务，如图像分类、语言模型训练等。
- **原理**：
  - 将数据集分割成多个子集，每个 GPU（或进程）处理一部分数据。
  - 每个 GPU 拥有完整的模型副本，独立计算梯度。
  - 通过**AllReduce**操作，在反向传播后同步所有 GPU 的梯度，更新模型参数。
  - 使用 Ring-AllReduce 算法优化通信效率，减少同步开销。
- **优点**：
  - 训练速度随 GPU 数量近线性提升。
  - 显存占用均衡，支持大模型训练。
  - 与单 GPU 代码修改量小，易于实现。
- **缺点**：
  - 需要多 GPU 或分布式环境支持。
  - 通信开销可能成为瓶颈，尤其在网络较慢时。

---

### DDP 的原理

1. **数据分片**：
   - 数据集被分成多个子集，每个 GPU（进程）处理一个子集（mini-batch）。
2. **模型复制**：
   - 每个 GPU 持有相同的模型副本，初始参数一致。
3. **并行计算**：
   - 每个 GPU 独立执行前向和反向传播，计算本地梯度。
4. **梯度同步**：
   - 使用 AllReduce 操作（基于 NCCL 或 MPI），将所有 GPU 的梯度求平均，确保参数更新一致。
5. **参数更新**：
   - 每个 GPU 使用平均梯度更新模型参数，保持模型同步。
6. **分布式初始化**：
   - 通过 `torch.distributed.init_process_group` 初始化通信后端（如 NCCL），确保进程间通信。

---

### 简单代码示例：基于 PyTorch 的 DDP 训练

以下是一个简单的例子，展示如何在 PyTorch 中使用 DDP 在多 GPU 上训练一个简单神经网络（以 MNIST 数据集为例）。代码假设运行在单机多 GPU 环境。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

# 1. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. 训练函数
def train(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 数据加载
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dataset)  # 分布式数据采样
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    
    # 模型和优化器
    model = SimpleNet().to(device)
    model = DDP(model, device_ids=[rank])  # 包装为 DDP 模型
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(2):  # 2 个 epoch 作为示例
        model.train()
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.6f}")
    
    # 清理
    dist.destroy_process_group()

# 3. 主函数
def main():
    world_size = torch.cuda.device_count()  # GPU 数量
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # 设置环境变量（单机多 GPU）
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    main()
```

---

### 运行方式
在命令行运行：
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS your_script.py
```
其中 `NUM_GPUS` 是可用 GPU 数量（如 2）。或者直接运行脚本（代码已包含 `mp.spawn`）。

---

### 代码说明

1. **模型定义**：
   - 定义一个简单的全连接网络，用于 MNIST 分类（输入 28x28，输出 10 类）。

2. **分布式初始化**：
   - 使用 `dist.init_process_group` 初始化分布式环境（`backend="nccl"` 用于 GPU）。
   - `MASTER_ADDR` 和 `MASTER_PORT` 设置通信地址（单机用 localhost）。

3. **数据加载**：
   - 使用 `DistributedSampler` 自动将数据集分片，每个 GPU 处理一部分数据。
   - `set_epoch` 确保每个 epoch 数据随机打乱。

4. **DDP 模型**：
   - `DDP(model, device_ids=[rank])` 将模型包装为分布式并行，自动同步梯度。
   - 每个 GPU 运行独立的前向/反向传播，DDP 负责梯度 AllReduce。

5. **训练与清理**：
   - 标准训练循环，计算损失并更新参数。
   - 训练结束后销毁进程组，释放资源。

---

### 关键点

1. **分布式采样**：
   - `DistributedSampler` 确保每个 GPU 处理不同数据子集，避免重复。
2. **梯度同步**：
   - DDP 自动在反向传播后同步梯度，保持模型一致。
3. **扩展性**：
   - 可结合 **AMP**（参考前文，加入 `torch.cuda.amp` 加速）、**Curriculum Learning**（逐步引入复杂数据）、**Optuna/Ray Tune**（优化超参数）。
   - 示例中可加入 `StandardScaler` 预处理数据（参考前文）。
4. **硬件要求**：
   - 需要多 GPU 或分布式集群，NCCL 后端优化 GPU 通信。

---

### 实际效果

- **训练速度**：随 GPU 数量增加，训练时间近线性减少（例如 2 个 GPU 接近 2 倍加速）。
- **显存分配**：每个 GPU 独立处理数据分片，显存占用均衡。
- **精度保持**：与单 GPU 训练精度一致，因梯度同步确保模型一致性。
- **适用性**：适合大规模模型（如 ResNet、Transformer），在多 GPU 或集群中效果显著。

