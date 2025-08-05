## 数据集简介
### UCI HAR数据集
UCI HAR（Human Activity Recognition）数据集是用于研究人类活动识别的公开数据集，由意大利热那亚大学的Davide Anguita等人于2012年收集并发布，托管在UCI机器学习数据库中。它通过智能手机（Samsung Galaxy SII）上的加速度计和陀螺仪传感器，记录了30名年龄在19至48岁的受试者在执行日常活动时的运动数据，广泛应用于时间序列数据分类和机器学习研究。[](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)[](https://archive.ics.uci.edu/ml/datasets/Human%2BActivity%2BRecognition%2BUsing%2BSmartphones)

### 数据集概述
- **目的**：用于开发和测试人类活动识别算法，特别是在基于传感器的时间序列数据分类任务中。
- **活动类别**：数据集包含6种基本活动：
  走路（Walking）
  上楼（Walking Upstairs）
  下楼（Walking Downstairs）
  坐着（Sitting）
  站立（Standing）
  躺下（Laying）
- **受试者**：30名受试者，数据分为训练集（70%，21人，7352个样本）和测试集（30%，9人，2947个样本），无受试者重叠。
- **传感器**：使用腰部佩戴的智能手机，采集以下传感器数据：
  - 三轴加速度计（x、y、z方向，包含重力加速度和身体运动分量）
  - 三轴陀螺仪（x、y、z方向的角速度）
- **数据采样**：
  - 采样频率：50 Hz（每秒50次采样）。
  - 数据以2.56秒的固定宽度滑动窗口（128个读数/窗口）进行采样，窗口间50%重叠。
  - 原始传感器信号通过巴特沃斯低通滤波器（Butterworth filter）预处理，分离身体加速度和重力加速度。
- **特征**：
  - 原始数据集包含9个时间序列信号（加速度计和陀螺仪各3轴，总加速度3轴）。
  - 提供了一个预处理的特征集，包含561个特征向量，通过对时间序列数据进行特征工程提取（如均值、标准差、频域特征等）。
- **数据规模**：
  - 训练集：7352个样本
  - 测试集：2947个样本
  - 总计：10,299个样本
- **许可**：数据集遵循Creative Commons Attribution 4.0 International (CC BY 4.0)许可，可自由使用和分享，但需注明出处。[](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)

### 数据集结构
数据集以压缩文件形式提供（约58 MB），解压后包含以下主要文件和目录：
- **train/**：训练集数据
  - `X_train.txt`：训练集特征数据（7352行，561列特征）。
  - `y_train.txt`：训练集活动标签（1-6，对应6种活动）。
  - `subject_train.txt`：训练集受试者ID（1-30）。
- **test/**：测试集数据
  - `X_test.txt`：测试集特征数据（2947行，561列特征）。
  - `y_test.txt`：测试集活动标签。
  - `subject_test.txt`：测试集受试者ID。
- **features.txt**：561个特征的名称列表。
- **activity_labels.txt**：活动标签（1-6）与活动名称的映射。
- **README.txt**：数据集的技术描述。[](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)

### 数据采集方式
- 受试者在腰部佩戴智能手机，执行6种活动，实验过程通过视频记录以进行手动标注。
- 传感器数据包括加速度计和陀螺仪的x、y、z轴信号，记录了身体运动和重力分量。
- 数据经过噪声滤波和预处理，生成适合机器学习的时间序列特征。

### 应用与研究
UCI HAR数据集广泛用于以下研究领域：
- **时间序列分类**：使用机器学习（如SVM、随机森林）或深度学习（如LSTM、CNN）进行活动分类。[](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)[](https://arxiv.org/html/2505.06730v1)
- **特征工程**：研究如何从原始时间序列中提取有效特征（如TSFresh工具）。[](https://medium.com/data-science/a-guide-to-time-series-sensor-data-classification-using-uci-har-data-b7ac4f6ad251)
- **跨域泛化**：测试模型在不同数据集或设备上的泛化能力。[](https://www.nature.com/articles/s41597-024-03951-4)
- **缺失数据处理**：探索在传感器数据缺失情况下的活动识别方法。[](https://arxiv.org/html/2505.06730v1)
- **个性化识别**：尝试识别执行活动的个体（受试者分类）。[](https://arxiv.org/html/2505.06730v1)

### 相关研究成果
- 数据集首次在2012年的论文《Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine》中用于测试SVM模型。[](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)
- 后续研究使用深度学习方法（如LSTM、CNN-LSTM），在活动分类任务中取得93%-99%的准确率。[](https://arxiv.org/html/2505.06730v1)[](https://www.researchgate.net/figure/Description-of-UCI-HAR-dataset_tbl1_349651970)
- 例如，ConvResBiGRU-SE模型在UCI HAR数据集上达到99.18%的准确率，展示了深度残差网络和注意力机制的优越性。[](https://www.researchgate.net/figure/Description-of-UCI-HAR-dataset_tbl1_349651970)

### 获取数据集
- **下载地址**：UCI机器学习数据库（https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones）
- **Kaggle**：也提供数据集，但建议使用UCI原始数据集以确保与文献中结果的一致性。[](https://www.researchgate.net/post/Can_Anyone_help_me_in_understandingc_features_in_UCI_HAR_Dataset)

### 注意事项
- **数据一致性**：Kaggle版本的数据可能与UCI原始数据集的分区不同，建议使用UCI官方版本以确保可比性。[](https://www.researchgate.net/post/Can_Anyone_help_me_in_understandingc_features_in_UCI_HAR_Dataset)
- **预处理**：原始时间序列数据需要额外处理（如标准化、缺失值填补）以适应特定模型。[](https://arxiv.org/html/2505.06730v1)[](https://www.nature.com/articles/s41597-024-03951-4)
- **挑战**：活动识别涉及大量传感器数据（每秒数十次观测），需要处理时间序列的复杂性和个体运动模式的差异。[](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)

