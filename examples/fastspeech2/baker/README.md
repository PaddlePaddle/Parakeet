

# FastSpeech2 with BZNSYP

## Dataset

### Download and Extract the datasaet.
Download BZNSYP from it's [Official Website](https://test.data-baker.com/data/index/source).
### Get MFA result of BZNSYP and Extract it.

We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
You can download from here [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), or train your own MFA model reference to  [use_mfa example](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/use_mfa) of our repo.

### Preprocess the dataset.

Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of BZNSYP is `./baker_alignment_tone`.
Run the command below to preprocess the dataset.

```bash
./preprocess.sh
```
## Train the model
```bash
./run.sh
```
## Synthesize
We use [parallel wavegan](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/parallelwave_gan/baker) as the neural vocoder.
Download pretrained parallel wavegan model from [parallel_wavegan_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/parallel_wavegan_baker_ckpt_0.4.zip) and unzip it.
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
Pretrained Model with no sil in the edge of audios can be downloaded here. [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_ckpt_0.4.zip)

Then, you can use the following scripts to synthesize for `sentences.txt` using pretrained fastspeech2 model.
```bash
python3 synthesize_e2e.py \
  --fastspeech2-config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
  --fastspeech2-checkpoint=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
  --fastspeech2-stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy \
  --pwg-config=parallel_wavegan_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-params=parallel_wavegan_baker_ckpt_0.4/pwg_generator.pdparams \
  --pwg-stat=parallel_wavegan_baker_ckpt_0.4/pwg_stats.npy \
  --text=sentences.txt \
  --output-dir=exp/debug/test_e2e \
  --device="gpu" \
  --phones-dict=fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
```
