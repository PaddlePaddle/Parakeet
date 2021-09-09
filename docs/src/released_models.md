# Released Models
TTS system mainly includes three modules: `text frontend`, `acoustic model` and `vocoder`. We introduce a rule based Chinese text frontend in [cn_text_frontend.md](./cn_text_frontend.md). Here, we will introduce acoustic models and vocoders, which are trainable models.

The main processes of TTS include:
1. Convert the original text into linguistic features, such as phonemes, through `text frontend` module.
2. Convert linguistic features into acoustic features , such as spectrogram, mel spectrum, lpc features, etc. through `acoustic models`.
3. Convert acoustic features into waveforms through `vocoders`.

A simple text frontend module can be implemented by rules. Acoustic models and vocoders need to be trained. The models provided by Parakeet are acoustic models and vocoders.

## Acoustic Models
### Modeling Objectives of Acoustic Models

Modeling the mapping relationship between text sequences and speech features：
```text
text X = {x1,...,xM}
specch Y = {y1,...yN}
```
Modeling Objectives:
```text
Ω = argmax p(Y|X,Ω)
```
### Modeling process of Acoustic Models

- Frame level acoustic model:
   - duration model (M Frame - > N frame)
   - N frame - > N frame

<div align="left">
  <img src="../images/frame_level_am.png" width=500 /> <br>
</div>

- Sequence to sequence acoustic model:
    - M Frame - > N frame

<div align="left">
  <img src="../images/seq2seq_am.png" width=500 /> <br>
</div>

### Tacotron2
 [Tacotron](https://arxiv.org/abs/1703.10135)  is the first end-to-end acoustic model based on deep learning, and it is also the most widely used acoustic model.

[Tacotron2](https://arxiv.org/abs/1712.05884) is the Improvement of Tacotron.
#### Tacotron
Structure of Tacotron:
- Encoder
   - CBHG
   - Input: character sequence
- Decoder
    - Global soft attention
    - unidirectional RNN
    - Autoregressive teacher force training (input real speech feature)
    - Multi frame prediction
    - CBHG postprocess
    - Vocoder: Griffin-Lim
<div align="left">
  <img src="../images/tacotron.png" width=500 /> <br>
</div>

Advantage of Tacotron:
- No need for complex text frontend analysis modules.
- No need for additional duration model.
- Greatly simplifies the acoustic model construction process and reduces the dependence of speech synthesis tasks on domain knowledge.

Problems with Tacotron:
- The CBHG  is complex and the amount of parameters is relatively large
- Global soft attention
- Poor stability for speech synthesis tasks
- In training, the less the number of speech frames predicted at each moment, the more difficult it is to train.
-  Phase problem in Griffin-Lim casues speech distortion during wave reconstruction
- The autoregressive decoder cannot be stopped during the generation process

#### Tacotron2
Improvements of Tacotron2 for Tacotron:
- Reduction of parameters
   - CBHG -> 3 Conv layers + BLSTM or 5 Conv layers
   - remove Attention RNN
- Speech distortion caused by Griffin-Lim
    - WaveNet
- Improvements of PostNet
   - CBHG -> 5 Conv layers
   -  The input and output of the PostNet calculate `L2` loss with real Mel spectrum.
   - Residual connection
- Bad stop in autoregressive decoder
   - Predict whether it should stop at each moment of decoding (stop token)
   - Set a threshold to determine whether to stop generating when decoding
- Stability of attention
   - Location sensitive attention (LSA)
   - The alignment matrix of previous time is considered at the step `t` of decoder

<div align="left">
  <img src="../images/tacotron2.png" width=500 /> <br>
</div>

You can find Parakeet's tacotron2 example at `Parakeet/examples/tacotron2`.

### TransformerTTS
Transformer TTS is a combination of Tacotron2 and [Transformer](https://arxiv.org/abs/1706.03762).

#### Shortcomings of the Tacotrons
- Encodr and decoder are relatively weak at global information modeling
   - Vanishing gradient of RNN
   - Fixed-length context modeling problem in CNN kernel
- Training is relatively inefficient
- The attention is not robust enough and the stability is poor


#### Transformer
- Sequence-to-sequence model based entirely on attention mechanism
- Encoder
    - N blocks based on self-attention mechanism
    - Positional Encoding
- Decoder
    - N blocks based on self-attention mechanism
    - Add Mask to the self-attention in blocks to cover up the information after `t` step
    - Attentions between encoder and decoder
    - Positional Encoding

<div align="left">
  <img src="../images/transformer.png" width=500 /> <br>
</div>

#### Transformer TTS
Transformer TTS is a sequence-to-sequence acoustic model based on Transformer and Tacotron2.

Motivations：
- RNNs in Tacotron2  make the inefficiency of training
- Vanishing gradient of RNN makes the model's ability to model long-term contexts weak
- Self-attention doesn't contain any recursive structure which can be trained in parallel
- Self-attention can model global context information well

Innovations of Transformer TTS:
- Add conv based Prenet in encoder and decoder
- Stop Token in decoder controls when to stop autoregressive generation
- Add Postnet after decoder to improve the quality of synthetic speech
- Scaled position encoding
    -     Uniform scale position encoding may have a negative impact on input or output sequences

<div align="left">
  <img src="../images/transformer_tts.png" width=500 /> <br>
</div>

### SpeedySpeech
### FastSpeech2
### FastSpeech
### FastPitch
### FastSpeech2




## Vocoders

### Parallel WaveGAN

### WaveFlow
