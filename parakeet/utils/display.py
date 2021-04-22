# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import matplotlib
import librosa
import librosa.display
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib import cm, pyplot

__all__ = [
    "plot_alignment",
    "plot_spectrogram",
    "plot_waveflow",
]



def plot_alignment(alignment, title=None):
    # alignment: [encoder_steps, decoder_steps)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if title is not None:
        xlabel += '\n\n' + title
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return fig


def plot_spectrogram(spec):
    # spec: [C, T] librosa convention
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    return fig


def plot_waveform(spec, sr=22050):
    # spec: [C, T] librosa convention
    fig, ax = plt.subplots(figsize=(12, 3))
    im = librosa.display.waveplot(y, sr=22050)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
