
# FastSpeech2 with AISHELL-3

## Introduction
[AISHELL-3](http://www.aishelltech.com/aishell_3) is a large-scale and high-fidelity multi-speaker Mandarin speech corpus which could be used to train multi-speaker Text-to-Speech (TTS) systems.
We use AISHELL-3 to train a multi-speaker fastspeech2 model here.

## Dataset

### Download and Extract the datasaet
Download AISHELL-3.
```bash
wget https://www.openslr.org/resources/93/data_aishell3.tgz
```
Extract AISHELL-3.
```bash
mkdir data_aishell3
tar zxvf data_aishell3.tgz -C data_aishell3
```

### Get MFA result of BZNSYP and Extract it

We use [MFA2.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for aishell3_fastspeech2.
You can download from here [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz), or train your own MFA model reference to [use_mfa example](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/use_mfa) (use MFA1.x now) of our repo.

### Preprocess the dataset

Assume the path to the dataset is `~/datasets/data_aishell3`.
Assume the path to the MFA result of AISHELL-3 is `./aishell3_alignment_tone`.
Run the command below to preprocess the dataset.

```bash
./preprocess.sh
```
## Train the model
```bash
./run.sh
```
If you want to train fastspeech2 with cpu, please add `--device=cpu` arguments for `python3 train.py` in `run.sh`.
## Synthesize
We use [parallel wavegan](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/parallelwave_gan/baker) as the neural vocoder.
Download pretrained parallel wavegan model (Trained with baker) from [fastspeech2_nosil_aishell3_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_aishell3_ckpt_0.4.zip) and unzip it.
```bash
unzip parallel_wavegan_baker_ckpt_0.4.zip
```
`synthesize.sh` can synthesize waveform from `metadata.jsonl`.
`synthesize_e2e.sh` can synthesize waveform from text list.

```bash
./synthesize.sh
```
or
```bash
./synthesize_e2e.sh
```

You can see the bash files for more datails of input parameters.

## Pretrained Model
Pretrained Model with no sil in the edge of audios can be downloaded here. [fastspeech2_nosil_aishell3_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_aishell3_ckpt_0.4.zip)

Then, you can use the following scripts to synthesize for `../sentences.txt` using pretrained fastspeech2 model.
```bash
python3 synthesize_e2e.py \
  --fastspeech2-config=fastspeech2_nosil_aishell3_ckpt_0.4/default.yaml \
  --fastspeech2-checkpoint=fastspeech2_nosil_aishell3_ckpt_0.4/snapshot_iter_96400.pdz \
  --fastspeech2-stat=fastspeech2_nosil_aishell3_ckpt_0.4/speech_stats.npy \
  --pwg-config=parallel_wavegan_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-params=parallel_wavegan_baker_ckpt_0.4/pwg_generator.pdparams \
  --pwg-stat=parallel_wavegan_baker_ckpt_0.4/pwg_stats.npy \
  --text=../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=fastspeech2_nosil_aishell3_ckpt_0.4/phone_id_map.txt \
  --speaker-dict=fastspeech2_nosil_aishell3_ckpt_0.4/speaker_id_map.txt

```
## Future work
A multi-speaker  vocoder is needed.
