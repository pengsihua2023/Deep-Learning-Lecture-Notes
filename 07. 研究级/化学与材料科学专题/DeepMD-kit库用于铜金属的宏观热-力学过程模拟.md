## DeepMD-kit库用于铜金属的宏观热-力学过程模拟
DeepMD-kit在材料科学中的一个典型真实数据例子是模拟铜（Cu）材料。该例子来源于一篇关于大规模分子动力学模拟的论文，使用ab initio（第一性原理）数据训练Deep Potential（DP）模型，然后在超算上进行高精度MD模拟，适用于研究金属材料的热力学性质、缺陷行为等。

### 1. 真实数据来源
- 数据来源于ab initio训练数据集，使用并发学习方案（concurrent learning scheme）生成，确保模型在广阔热力学区域内具有均匀精度。
- 数据包括铜原子的局部环境描述、能量和力等，基于量子力学计算（如DFT），覆盖相关构型空间。

### 2. 模型训练过程
- 使用DeePMD-kit包实现训练，通常在单个GPU卡上耗时几小时到一周，取决于数据复杂度。
- 模型采用深度神经网络（DNN）拟合高维函数，融入对称性约束和并发学习，以最小化数据集需求。
- 命令示例（假设数据目录为`cu_data`）：
  ```
  dp train input.json
  ```
- 输入配置文件`input.json`示例（简化版，适用于铜系统，使用se_a描述符）：
  ```json
  {
      "model": {
          "type_map": ["Cu"],
          "descriptor": {
              "type": "se_a",
              "rcut": 8.0,  // 截断半径8 Å
              "rcut_smth": 0.5,
              "sel": [512],  // 邻居原子数上限512
              "neuron": [32, 64, 128]  // 嵌入网络大小
          },
          "fitting_net": {
              "neuron": [240, 240, 240],  // 拟合网络大小
              "resnet_dt": true
          }
      },
      "learning_rate": {
          "type": "exp",
          "start_lr": 0.001,
          "stop_lr": 1e-8
      },
      "loss": {
          "start_pref_e": 0.1,
          "limit_pref_e": 1,
          "start_pref_f": 1000,
          "limit_pref_f": 1
      },
      "training": {
          "systems": ["cu_data"],
          "batch_size": "auto",
          "numb_steps": 1000000
      }
  }
  ```
- 训练后冻结模型：
  ```
  dp freeze -o frozen.pb
  ```

### 3. MD模拟细节
- 系统规模：可扩展至1.274亿原子（127,401,984 atoms），使用弱缩放测试从796万原子到1.27亿原子。
- 模拟设置：
  - 时间步长：1.0 fs。
  - 积分方案：velocity-Verlet。
  - 初始温度：330 K（Boltzmann分布）。
  - 邻居列表更新：每50步，缓冲区2 Å。
  - 运行步数：示例中为500步，但可扩展至纳秒级。
- 性能（在Summit超算上，4560节点）：
  - 双精度：91 PFLOPS，时间/步/原子：8.1 × 10⁻¹⁰ s。
  - 混合单精度：162 PFLOPS，时间/步/原子：4.6 × 10⁻¹⁰ s。
  - 混合半精度：275 PFLOPS，时间/步/原子：2.7 × 10⁻¹⁰ s。
  - 对于1.27亿原子系统，纳秒模拟可在29小时（双精度）内完成。
- LAMMPS集成示例脚本（in.lammps，假设冻结模型为`frozen.pb`）：
  ```
  units metal
  boundary p p p
  atom_style atomic

  read_data cu_conf.lmp  # 初始铜晶体配置文件

  pair_style deepmd frozen.pb
  pair_coeff * *

  mass 1 63.546  # Cu原子质量

  timestep 0.001
  thermo 50

  fix 1 all nvt temp 330.0 330.0 0.1
  run 500  # 示例运行500步
  ```
- 运行命令（需安装LAMMPS的DeepMD插件）：
  ```
  mpirun -np 4 lmp -i in.lammps
  ```

此例子展示了DeepMD-kit在材料科学中处理金属系统的高效性，可扩展到其他材料如合金或半导体。更多细节可参考DeepMD-kit的GitHub示例文件夹。
