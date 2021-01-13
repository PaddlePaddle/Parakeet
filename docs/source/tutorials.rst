===========
Tutorials
===========

Basic Usage
-------------------

Pretrained models are provided in a archive. Extract it to get a folder like this::

    checkpoint_name/
    ├──config.yaml
    └──step-310000.pdparams

The ``config.yaml`` stores the config used to train the model, the ``step-N.pdparams`` is the parameter file, where N is the steps it has been trained.

The example code below shows how to use the models for prediction.

text to spectrogram
^^^^^^^^^^^^^^^^^^^^^^

The code below show how to use a transformer_tts model. After loading the pretrained model, use ``model.predict(sentence)`` to generate spectrogram (in numpy.ndarray format), which can be further used to synthesize waveflow.

>>> import parakeet
>>> from parakeet.frontend import English
>>> from parakeet.models import TransformerTTS
>>> from pathlib import Path
>>> import yacs
>>> 
>>> # load the pretrained model
>>> frontend = English()
>>> checkpoint_dir = Path("transformer_tts_pretrained")
>>> config = yacs.config.CfgNode.load_cfg(str(checkpoint_dir / "config.yaml"))
>>> checkpoint_path = str(checkpoint_dir / "step-310000")
>>> model = TransformerTTS.from_pretrained(
>>>     frontend, config, checkpoint_path)
>>> model.eval()
>>> 
>>> # text to spectrogram
>>> sentence = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
>>> outputs = model.predict(sentence, verbose=args.verbose)
>>> mel_output = outputs["mel_output"]

vocoder
^^^^^^^^^^

Like the example above, after loading the pretrained ConditionalWaveFlow model, call ``model.predict(mel)`` to synthesize waveflow (in numpy.ndarray format).

>>> import soundfile as df
>>> from parakeet.models import ConditionalWaveFlow
>>> 
>>> # load the pretrained model
>>> checkpoint_dir = Path("waveflow_pretrained")
>>> config = yacs.config.CfgNode.load_cfg(str(checkpoint_dir / "config.yaml"))
>>> checkpoint_path = str(checkpoint_dir / "step-2000000")
>>> vocoder = ConditionalWaveFlow.from_pretrained(config, checkpoint_path)
>>> vocoder.eval()
>>> 
>>> # synthesize
>>> audio = vocoder.predict(mel_output)
>>> sf.write(audio_path, audio, config.data.sample_rate)

For more details on how to use the model, please refer the documentation.





