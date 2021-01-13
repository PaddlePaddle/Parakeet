# Tacotron2

PaddlePaddle dynamic graph implementation of Tacotron2, a neural network architecture for speech synthesis directly from text. The implementation is based on [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884).

## Project Structure

```text
├── config.py              # default configuration file
├── ljspeech.py            # dataset and dataloader settings for LJSpeech
├── preprocess.py          # script to preprocess LJSpeech dataset
├── synthesis.py           # script to synthesize spectrogram from text
├── train.py               # script for tacotron2 model training
```

## Dataset

We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

Then you need to preprocess the data by running ``preprocess.py``, the preprocessed data will be placed in ``--output`` directory.

```bash
python preprocess.py \
--input=${DATAPATH} \
--output=${PREPROCESSEDDATAPATH} \
-v  \
```

For more help on arguments

``python preprocess.py --help``.

## Train the model

Tacotron2 model can be trained by running ``train.py``.

```bash
python train.py \
--data=${PREPROCESSEDDATAPATH} \
--output=${OUTPUTPATH} \
--device=gpu \
```

If you want to train on CPU, just set ``--device=cpu``.
If you want to train on multiple GPUs, just set ``--nprocs`` as num of GPU.
By default, training will be resumed from the latest checkpoint in ``--output``, if you want to start a new training, please use a new ``${OUTPUTPATH}`` with no checkpoint. And if you want to resume from an other existing model, you should set ``checkpoint_path`` to be the checkpoint path you want to load.

**Note: The checkpoint path cannot contain the file extension.**

For more help on arguments

``python train_transformer.py --help``.

## Synthesis

After training the Tacotron2, spectrogram can be synthesized by running ``synthesis.py``.

```bash
python synthesis.py \
--config=${CONFIGPATH} \
--checkpoint_path=${CHECKPOINTPATH} \
--input=${TEXTPATH} \
--output=${OUTPUTPATH}
--device=gpu
```

The ``${CONFIGPATH}`` needs to be matched with ``${CHECKPOINTPATH}``.

For more help on arguments

``python synthesis.py --help``.

Then you can find the spectrogram files in ``${OUTPUTPATH}``, and then they can be the input of vocoder like [waveflow](../waveflow/README.md#Synthesis) to get audio files.
