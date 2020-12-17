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
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib import cm, pyplot

__all__ = [
    "pack_attention_images",
    "add_attention_plots",
    "plot_alignment",
    "min_max_normalize",
    "add_spectrogram_plots",
]


def pack_attention_images(attention_weights, rotate=False):
    # add a box
    attention_weights = np.pad(attention_weights, [(0, 0), (1, 1), (1, 1)],
                               mode="constant",
                               constant_values=1.)
    if rotate:
        attention_weights = np.rot90(attention_weights, axes=(1, 2))
    n, h, w = attention_weights.shape

    ratio = h / w
    if ratio < 1:
        cols = max(int(np.sqrt(n / ratio)), 1)
        rows = int(np.ceil(n / cols))
    else:
        rows = max(int(np.sqrt(n / ratio)), 1)
        cols = int(np.ceil(n / rows))
    extras = rows * cols - n
    #print(rows, cols, extras)
    total = np.append(attention_weights, np.zeros([extras, h, w]), axis=0)
    total = np.reshape(total, [rows, cols, h, w])
    img = np.block([[total[i, j] for j in range(cols)] for i in range(rows)])
    return img


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return data


def plot_alignment(alignment, title=None):
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

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def add_attention_plots(writer, tag, attention_weights, global_step):
    img = plot_alignment(attention_weights.numpy().T)
    writer.add_image(tag, img, global_step, dataformats="HWC")


def add_multi_attention_plots(writer, tag, attention_weights, global_step):
    attns = [attn[0].numpy() for attn in attention_weights]
    for i, attn in enumerate(attns):
        img = pack_attention_images(attn)
        writer.add_image(
            f"{tag}/{i}",
            cm.plasma(img),
            global_step=global_step,
            dataformats="HWC")


def add_spectrogram_plots(writer, tag, spec, global_step):
    spec = spec.numpy().T
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    writer.add_image(tag, data, global_step, dataformats="HWC")


def min_max_normalize(v):
    return (v - v.min()) / (v.max() - v.min())
