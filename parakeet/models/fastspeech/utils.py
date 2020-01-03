import numpy as np

def get_alignment(attn_probs, n_head):
    max_F = 0
    assert attn_probs[0].shape[0] % n_head == 0
    batch_size = int(attn_probs[0].shape[0] // n_head)
    for i in range(len(attn_probs)):
        multi_attn = attn_probs[i].numpy()
        for j in range(n_head):
            attn = multi_attn[j*batch_size:(j+1)*batch_size]
            F = score_F(attn)
            if max_F < F:
                max_F = F
                max_attn = attn
    alignment = compute_duration(max_attn)
    return alignment
    
def score_F(attn):
    max = np.max(attn, axis=-1)
    mean = np.mean(max)
    return mean

def compute_duration(attn):
    alignment = np.zeros([attn.shape[0],attn.shape[2]])
    for i in range(attn.shape[0]):
        for j in range(attn.shape[1]):
            max_index = attn[i,j].tolist().index(attn[i,j].max())
            alignment[i,max_index] += 1

    return alignment


