## DeepMD-kit库用于分子动力学模拟
以下是基于DeepMD-kit的例子代码实现，使用真实数据（以气相甲烷分子为例）。这个例子来源于DeepMD-kit的官方教程，使用从VASP生成的ab-initio分子动力学数据（OUTCAR文件）作为真实数据来源。过程包括数据准备、模型训练、冻结、测试等步骤。

### 1. 数据准备
首先，需要下载并解压甲烷数据（真实ab-initio轨迹数据）。然后，使用Python代码将数据转换为DeepMD-kit格式，并拆分为训练集和验证集。

#### 下载数据命令：
```
wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/CH4.tar
tar xvf CH4.tar
```

#### Python代码（数据准备，使用dpdata包）：
```python
import dpdata 
import numpy as np

# 加载VASP OUTCAR文件中的数据（真实ab-initio MD轨迹）
data = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar') 
print('# the data contains %d frames' % len(data))

# 随机选择40个帧作为验证数据
index_validation = np.random.choice(200, size=40, replace=False)     
# 剩余的作为训练数据
index_training = list(set(range(200)) - set(index_validation))       
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

# 将训练数据导出到目录 "training_data"
data_training.to_deepmd_npy('training_data')               

# 将验证数据导出到目录 "validation_data"
data_validation.to_deepmd_npy('validation_data')           

print('# the training data contains %d frames' % len(data_training)) 
print('# the validation data contains %d frames' % len(data_validation)) 
```
**说明**：数据包含原子类型、坐标、力、能量等，共200帧。训练集约160帧，验证集40帧。导出后，目录中会生成如`box.npy`、`coord.npy`、`energy.npy`、`force.npy`等文件。

### 2. 模型配置（JSON输入文件：input.json）
创建一个名为`input.json`的文件，内容如下（使用DeepPot-SE描述符，适用于甲烷系统）：
```json
{
    "model": {
        "type_map": ["H", "C"],                           
        "descriptor": {
            "type": "se_e2_a",                    
            "rcut": 6.00,                        
            "rcut_smth": 0.50,                         
            "sel": [4, 1],                       
            "neuron": [10, 20, 40],                 
            "resnet_dt": false,
            "axis_neuron": 4,                            
            "seed": 1,
            "_comment": "that's all"
        },
        "fitting_net": {
            "neuron": [100, 100, 100],   
            "resnet_dt": true,
            "seed": 1,
            "_comment": "that's all"
        },
        "_comment": "that's all"
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.001,    
        "stop_lr": 3.51e-8,
        "_comment": "that's all"
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment": "that's all"
    },
    "training": {
        "training_data": {
            "systems": ["../00.data/training_data"],     
            "batch_size": "auto",                       
            "_comment": "that's all"
        },
        "validation_data": {
            "systems": ["../00.data/validation_data/"],
            "batch_size": "auto",               
            "numb_btch": 1,
            "_comment": "that's all"
        },
        "numb_steps": 100000,                           
        "seed": 10,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "save_freq": 10000
    }
}
```
**说明**：`type_map`指定原子类型（H和C）；描述符使用`se_e2_a`类型；训练步数为100000。

### 3. 训练模型
使用以下命令启动训练：
```
dp train input.json
```
如果需要从检查点重启：
```
dp train --restart model.ckpt input.json
```
**说明**：训练会生成`lcurve.out`文件记录损失曲线。

### 4. 冻结和压缩模型
冻结模型：
```
dp freeze -o graph.pb
```
压缩模型（可选，提高效率）：
```
dp compress -i graph.pb -o graph-compress.pb
```


### 5. 测试模型
使用压缩后的模型测试验证数据：
```
dp test -m graph-compress.pb -s ../00.data/validation_data -n 40 -d results
```
**说明**：`-n 40`指定测试40帧数据，输出如能量和力的RMSE/MAE指标。

### 6. 可视化学习曲线（可选Python代码）
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("lcurve.out", names=True)
for name in data.dtype.names[1:-1]:
    plt.plot(data['step'], data[name], label=name)
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
```
**说明**：绘制训练过程中的损失曲线。

此例子基于真实ab-initio数据，可直接运行（需安装DeepMD-kit和dpdata）。如果需要水分子或其他系统的例子，可参考DeepMD-kit GitHub的`/examples`文件夹。
