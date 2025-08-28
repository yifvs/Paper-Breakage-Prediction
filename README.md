# Paper-Breakage-Prediction

本项目借鉴https://github.com/cran2367/autoencoder_classifier/blob/master/autoencoder_classifier.ipynb 中的思路，将原始的 Keras/TensorFlow，移植为可运行的 PyTorch 脚本。

## 当前脚本存在的问题

❌ 1. 特征工程不足
使用 EfficientFCParameters() 提取特征，虽然效率高，但可能丢失了关键的断裂相关特征。
FFT 特征计算中出现 NaN，可能导致部分重要信息被丢弃。
❌ 2. 数据预处理问题
curve_shift(df, shift_by=-2) 可能导致标签扩展不充分或引入噪声。
时间序列长度不足或存在缺失值，影响 tsfresh 特征提取质量。
❌ 3. 模型结构简单
自编码器结构较浅（仅两层编码/解码），可能不足以学习复杂的时间序列模式。
编码维度（32）可能过小，导致信息压缩过度。
❌ 4. 训练策略问题
仅用 y=0（正常样本）训练自编码器，未利用异常样本进行监督学习。
重构误差作为异常检测依据，但若正常样本本身波动大，会导致阈值难以设定。

## 后续改进计划

使用更深的自编码器或变分自编码器（VAE）
引入注意力机制或 LSTM 捕捉长期依赖
在训练阶段加入少量异常样本进行半监督学习
使用 Reconstruction Error + Anomaly Score 的联合损失函数
对重构误差进行标准化或归一化处理

