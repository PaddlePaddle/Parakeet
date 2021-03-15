import numpy as np
import paddle
import paddle.nn.functional as F


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.shape) - 1
    m = paddle.max(x, axis=axis)
    m2 = paddle.max(x, axis=axis, keepdim=True)
    return m + paddle.log(paddle.sum(paddle.exp(x - m2), axis=axis))


# It is adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
def discretized_mix_logistic_loss(y_hat, y, num_classes=65536,
                                  log_scale_min=None, reduce=True):
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))

    y_hat = y_hat.transpose([0, 2, 1])
    assert y_hat.dim() == 3
    assert y_hat.shape[1] % 3 == 0
    nr_mix = y_hat.shape[1] // 3

    # (B x T x C)
    y_hat = y_hat.transpose([0, 2, 1])

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = paddle.clip(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)
    centered_y = paddle.cast(y, 'float32') - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = F.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    # softplus: log(1+ e^{-x})
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = cdf_delta > 1e-5
    inner_inner_cond = paddle.cast(inner_inner_cond, dtype='float32')

    inner_inner_out = inner_inner_cond * \
                      paddle.log(paddle.clip(cdf_delta, min=1e-12)) + \
                      (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))

    inner_cond = y > 0.999
    inner_cond = paddle.cast(inner_cond, dtype='float32')

    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = y < -0.999
    cond = paddle.cast(cond, dtype='float32')

    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -paddle.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): (B, C, T)
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))

    assert y.shape[1] % 3 == 0
    nr_mix = y.shape[1] // 3


    # (B, T, C)
    y = y.transpose([0, 2, 1])
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = paddle.uniform(logit_probs.shape, dtype=logit_probs.dtype, min=1e-5, max=1.0 - 1e-5)
    temp = logit_probs - paddle.log(-paddle.log(temp))
    argmax = paddle.argmax(temp, axis=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix)
    one_hot = paddle.cast(one_hot, dtype='float32')

    # select logistic parameters
    means = paddle.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, axis=-1)
    log_scales = paddle.clip(paddle.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, axis=-1), min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = paddle.uniform(means.shape, min=1e-5, max=1.0 - 1e-5)
    x = means + paddle.exp(log_scales) * (paddle.log(u) - paddle.log(1. - u))
    x = paddle.clip(paddle.clip(x, min=-1.), max=1.)

    return x


if __name__ == '__main__':
    y_hat = paddle.rand([4, 5, 30])
    y = paddle.rand([y_hat.shape[0], y_hat.shape[1], 1]) * 65535
    y = paddle.cast(y, 'int32')
    print(discretized_mix_logistic_loss(y_hat, y))

    y = paddle.rand([1, 30, 4]) * 2456
    print(sample_from_discretized_mix_logistic(y))