
# FastSpeech2 with BZNSYP
------
## Dataset
-----
### Download and Extract the datasaet.
Download BZNSYP from it's [Official Website](https://test.data-baker.com/data/index/source).
### Get MFA result of BZNSYP and Extract it.

we use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
you can download from here, or train your own MFA model reference to  [use_mfa example](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/use_mfa) of our repo.

### Preprocess the dataset.

Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of BZNSYP is `./baker_alignment_tone`.
Run the command below to preprocess the dataset.

```bash
./preprocess.sh
```
## Train the model
---
```bash
./run.sh
```
## Synthesize
---
we use [parallel wavegan](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/parallelwave_gan/baker) as the neural vocoder.
`synthesize.sh` can synthesize waveform for `metadata.jsonl`.
`synthesize_e2e.sh` can synthesize waveform for text list.
```bash
./synthesize.sh
```
or
```bash
./synthesize_e2e.sh
```

you can see the bash files for more datails of input parameter.

## Pretrained Model
