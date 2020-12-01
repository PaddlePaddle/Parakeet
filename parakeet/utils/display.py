import numpy as np
import matplotlib
from matplotlib import cm, pyplot

def pack_attention_images(attention_weights, rotate=False):
    # add a box
    attention_weights = np.pad(attention_weights, 
                               [(0, 0), (1, 1), (1, 1)], 
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

def add_attention_plots(writer, tag, attention_weights, global_step):
    attns = [attn[0].numpy() for attn in attention_weights]
    for i, attn in enumerate(attns):
        img = pack_attention_images(attn)
        writer.add_image(f"{tag}/{i}", 
                        cm.plasma(img),
                        global_step=global_step,
                        dataformats="HWC")

def min_max_normalize(v):
    return (v - v.min()) / (v.max() - v.min())
