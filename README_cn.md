# Parakeet

Parakeet 自在为开源社区提供一个灵活，高效，先进的语音合成工具箱。Parakeet 基于 PaddlePaddle 2.0 构建，并且包含了 [百度研究院]((http://research.baidu.com)) 以及其他研究机构的许多有影响力的 TTS 模型。

<img src="./images/logo.png" alt="parakeet-logo" style="zoom: 33%;" />

其中包含了百度研究院最近提出的 [WaveFlow](https://arxiv.org/abs/1912.01219) 模型。

- WaveFlow 无需专用于推理的 kernel, 就可以在 Nvidia v100 上以 40 倍实时的速度合成 22.05kHz 的高保真度的语音。这比 [WaveGlow](https://github.com/NVIDIA/waveglow) 模型更快，而且比 WaveNet 快几个数量级。
- WaveFlow 是占用小的，基于流的用于生成原始音频的模型，只有 5.9M 个可训练参数，约为 WaveGlow (87.9M 个参数) 的 1/15.
- WaveFlow 可以直接通过最大似然方式训练，而不需要概率密度蒸馏，或者是类似 ParallelWaveNet 和 ClariNet 中使用的辅助 loss, 这简化了训练流程，减小了开发成本。

## 模型概览

为了方便使用已有的 TTS 模型以及开发新的模型，Parakeet 选取了经典的模型，并且提供了基于 PaddlePaddle 的参考实现。Parakeet 进一步抽象了 TTS 任务的流程，并且将数据预处理，模块共享，模型配置以及训练和合成的流程标准化。目前已经支持的模型包括音码器 (vocoder) 和声学模型。

- 音码器
  - [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)
  - [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281)
  - [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

- 声学模型
  - [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)
  - [Neural Speech Synthesis with Transformer Network (Transformer TTS)](https://arxiv.org/abs/1809.08895)
  - [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)

未来将会添加更多的模型。

如若需要基于 Parakeet 实现自己的模型和实验，可以参考 [如何准备自己的实验](./docs/experiment_guide_cn.md).

## 安装

请参考 [安装](./docs/installation_cn.md).

## 实验样例

Parakeet 提供了多个实验样例。这些样例使用 parakeet 中提供的模型，提供在公共数据集上进行实验的完整流程，包含数据处理，模型训练以及预测的功能，是进行实验以及二次开发的示例。

- [>>> WaveFlow](./examples/waveflow)
- [>>> Clarinet](./examples/clarinet)
- [>>> WaveNet](./examples/wavenet)
- [>>> Deep Voice 3](./examples/deepvoice3)
- [>>> Transformer TTS](./examples/transformer_tts)
- [>>> FastSpeech](./examples/fastspeech)


## 预训练模型和音频样例

Parakeet 同时提供了示例模型的训练好的参数，可从下表中获取。每一列列出了一个模型的资源，包含预训练模型的 checkpoint 下载 url, 训练该模型用的数据集，以及使用改 checkpoint 合成的语音样例。点击模型名，可以下载到一个压缩包，其中包含了训练该模型时使用的配置文件。

#### 音码器

我们提供了 residual channel 为 64, 96, 128 的 WaveFlow 模型 checkpoint. 另外还提供了 ClariNet 和 WaveNet 的 checkpoint.

<div align="center">
<table>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 64)</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 96)</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 128)</a>
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_ckpt_1.0.zip">ClariNet</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_ckpt_1.0.zip">WaveNet</a>
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>  
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>  
            </th>
        </tr>
    </tbody>
</table>
</div>


**注意:** 输入的 mel 频谱是从验证集中选取的，它们不被用于训练。

#### 声学模型

我们也提供了几个端到端的 TTS 模型的 checkpoint, 并展示用随机选取的著名引言合成的语音。对应的转录文本展示如下。

|   |Text| From |
|:-:|:-- | :--: |
0|*Life was like a box of chocolates, you never know what you're gonna get.* | *Forrest Gump* |  
1|*With great power there must come great responsibility.* | *Spider-Man*|
2|*To be or not to be, that’s a question.*|*Hamlet*|
3|*Death is just a part of life, something we're all destined to do.*| *Forrest Gump*|
4|*Don’t argue with the people of strong determination, because they may change the fact!*| *William Shakespeare* |

用于可以使用不同的音码器将声学模型产生的频谱转化为原始音频。我们将展示声学模型配合 [Griffin-Lim](https://ieeexplore.ieee.org/document/1164317) 音码器以及基于神经网络的音码器的合成样例。

##### 1) Griffin-Lim 音码器

<div align="center">
<table>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_1.0.zip">Transformer TTS</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_ckpt_1.0.zip">FastSpeech</a>
            </th>
                    </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th >
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th >
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
        </tr>
    </tbody>
    <thead>
</table>
</div>

##### 2) 神经网络音码器

正在开发中。

## 版权和许可

Parakeet 以 [Apache-2.0 license](LICENSE) 提供。
