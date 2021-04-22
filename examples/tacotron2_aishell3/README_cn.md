## Tacotron2 + AiShell3 数据集训练语音克隆模型

本实验的内容是利用 AiShell3 数据集和 Tacotron 2 模型进行语音克隆任务，使用的模型大体结构和论文 [Transfer Learning from Speaker Veriﬁcation to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) 相同。大致步骤如下：

1. Speaker Encoder: 我们使用了一个 Speaker Verification 任务训练一个 speaker encoder。这部分任务所用的数据集和训练 Tacotron 2 的数据集不同，因为不需要 transcription 的缘故，我们使用了较多的训练数据，可以参考实现 [ge2e](../ge2e)。
2. Synthesizer: 然后使用训练好的 encoder 为 AiShell3 数据集中的每个句子生成对应的 utterance embedding. 这个 Embedding 作为 Tacotron 模型中的一个额外输入和 encoder outputs 拼接在一起。
3. Vocoder: 我们使用的声码器是 WaveFlow，参考实验 [waveflow](../waveflow).

## 数据处理

### 音频处理

### 转录文本处理

### mel 频谱提取



## 训练



## 使用

参考 notebook 上的使用说明.