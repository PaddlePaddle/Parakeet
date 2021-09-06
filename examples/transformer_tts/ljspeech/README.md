# TransformerTTS with LJSpeech

## Dataset

### Download the datasaet
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```
### Extract the dataset
```bash
tar xjvf LJSpeech-1.1.tar.bz2
```

### Preprocess the dataset

Assume the path to the dataset is `~/datasets/LJSpeech-1.1`.
Run the command below to preprocess the dataset.

```bash
./preprocess.sh
```
## Train the model
```bash
./run.sh
```
If you want to train transformer_tts with cpu, please add `--device=cpu` arguments for `python3 train.py` in `run.sh`.
## Synthesize
We use [waveflow](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/waveflow) as the neural vocoder.
Download Pretrained WaveFlow Model with residual channel equals 128 from [waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip) and unzip it.
```bash
unzip waveflow_ljspeech_ckpt_0.3.zip
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
Pretrained Model can be downloaded here. [transformer_tts_ljspeech_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.4.zip)

Then, you can use the following scripts to synthesize for `../sentences.txt` using pretrained transformer_tts model.
```bash
python3 synthesize_e2e.py \
  --transformer-tts-config=transformer_tts_ljspeech_ckpt_0.4/default.yaml \
  --transformer-tts-checkpoint=transformer_tts_ljspeech_ckpt_0.4/snapshot_iter_201500.pdz \
  --transformer-tts-stat=transformer_tts_ljspeech_ckpt_0.4/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --text=../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=transformer_tts_ljspeech_ckpt_0.4/phone_id_map.txt
```
