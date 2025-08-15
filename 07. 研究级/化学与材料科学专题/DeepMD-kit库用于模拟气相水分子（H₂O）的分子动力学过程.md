## DeepMD-kit库用于模拟气相水分子（H₂O）的分子动力学过程
DeepMD-kit在量子化学中的一个典型真实数据例子是模拟气相水分子（H₂O）。该例子基于ab initio分子动力学（AIMD）数据，使用密度泛函理论（DFT）计算生成（如通过VASP或ABACUS软件），训练Deep Potential（DP）模型，以实现量子化学精度的高效模拟。数据包括原子坐标、力、能量等，覆盖水分子的振动和旋转模式，常用于研究氢键动力学或反应路径。

### 1. 真实数据来源
- 数据来源于ab initio计算的轨迹，例如使用VASP的OUTCAR文件生成的水分子AIMD模拟（约200帧，温度300 K）。这确保了量子化学精度（DFT水平，如PBE泛函）。
- 下载示例数据命令（如果可用，或从教程中准备）：
  ```
  wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/H2O.tar
  tar xvf H2O.tar
  ```
- 数据包含：原子类型（O和H）、坐标、力、能量和维里尔张量。

### 2. 数据准备
使用dpdata包将ab initio数据转换为DeepMD-kit格式，并拆分为训练集（约160帧）和验证集（40帧）。

#### Python代码（数据准备）：
```python
import dpdata
import numpy as np

# 加载VASP OUTCAR文件中的ab initio数据
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
**说明**：输出文件包括`box.npy`、`coord.npy`、`energy.npy`、`force.npy`等。原子类型文件`type.raw`示例：`0 1 1`（O为0，两个H为1）。

### 3. 模型配置（JSON输入文件：input.json）
创建一个名为`input.json`的文件，适用于水分子系统，使用DeepPot-SE描述符（se_e2_a类型）：
```json
{
    "model": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.00,
            "rcut_smth": 0.50,
            "sel": [2, 1],
            "neuron": [25, 50, 100],
            "resnet_dt": false,
            "axis_neuron": 16,
            "seed": 1,
            "_comment": "that's all"
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
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
            "systems": ["../training_data"],
            "batch_size": "auto",
            "_comment": "that's all"
        },
        "validation_data": {
            "systems": ["../validation_data"],
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
**说明**：`type_map`指定原子类型（O和H）；训练步数为100000，损失函数聚焦能量和力。

### 4. 训练模型
使用以下命令启动训练：
```
dp train input.json
```
如果需要重启：
```
dp train --restart model.ckpt input.json
```
**说明**：生成`lcurve.out`文件记录损失曲线（如能量误差~1 meV，力误差~100 meV/Å）。

### 5. 冻结和压缩模型
冻结模型：
```
dp freeze -o graph.pb
```
压缩模型（可选，提高效率）：
```
dp compress -i graph.pb -o graph-compress.pb
```

### 6. 测试模型
使用压缩模型测试验证数据：
```
dp test -m graph-compress.pb -s ../validation_data -n 40 -d results
```
**说明**：输出RMSE/MAE指标，验证量子化学精度。

### 7. 分子动力学模拟（集成LAMMPS）
使用训练模型进行AIMD代理模拟。LAMMPS输入脚本`in.lammps`示例（NVT系综，温度300 K）：
```
units           metal
boundary        p p p
atom_style      atomic

read_data       conf.lmp  # 初始水分子配置

pair_style      deepmd graph.pb
pair_coeff      * *

mass            1 15.999   # O原子质量
mass            2 1.00794  # H原子质量

timestep        0.001
thermo          100

fix             1 all nvt temp 300.0 300.0 0.1
run             10000

dump            1 all custom 100 h2o.dump id type x y z
```
初始`conf.lmp`示例：
```
3 atoms
2 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 15.999  # O
2 1.00794 # H

Atoms

1 1 0.000000 0.000000 0.000000
2 2 0.757000 0.586000 0.000000
3 2 -0.757000 0.586000 0.000000
```
运行命令：
```
lmp -i in.lammps
```
**说明**：生成轨迹`h2o.dump`，可用于分析水分子振动模式，与量子化学计算一致。

此例子可扩展到更大量子化学系统，如溶剂化效应或反应模拟。更多细节见DeepMD-kit GitHub的examples文件夹。
