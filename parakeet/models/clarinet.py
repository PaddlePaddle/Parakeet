import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle import distribution as D

from parakeet.models.wavenet import WaveNet, UpsampleNet, crop

class ParallelWaveNet(nn.LayerList):
    def __init__(self, n_loops, n_layers, residual_channels, condition_dim,
                 filter_size):
        """ParallelWaveNet, an inverse autoregressive flow model, it contains several flows(WaveNets).

        Args:
            n_loops (List[int]): `n_loop` for each flow.
            n_layers (List[int]): `n_layer` for each flow.
            residual_channels (int): `residual_channels` for every flow.
            condition_dim (int): `condition_dim` for every flow.
            filter_size (int): `filter_size` for every flow.
        """
        super(ParallelWaveNet, self).__init__()
        for n_loop, n_layer in zip(n_loops, n_layers):
            # teacher's log_scale_min does not matter herem, -100 is a dummy value
            self.append(
                WaveNet(n_loop, n_layer, residual_channels, 3, condition_dim,
                        filter_size, "mog", -100.0))

    def forward(self, z, condition=None):
        """Transform a random noise sampled from a standard Gaussian distribution into sample from the target distribution. And output the mean and log standard deviation of the output distribution.

        Args:
            z (Variable): shape(B, T), random noise sampled from a standard gaussian disribution.
            condition (Variable, optional): shape(B, F, T), dtype float, the upsampled condition. Defaults to None.

        Returns:
            (z, out_mu, out_log_std)
            z (Variable): shape(B, T), dtype float, transformed noise, it is the synthesized waveform.
            out_mu (Variable): shape(B, T), dtype float, means of the output distributions.
            out_log_std (Variable): shape(B, T), dtype float, log standard deviations of the output distributions.
        """
        for i, flow in enumerate(self):
            theta = flow(z, condition)  # w, mu, log_std [0: T]
            w, mu, log_std = paddle.chunk(theta, 3, axis=-1)  # (B, T, 1) for each
            mu = paddle.squeeze(mu, -1)  #[0: T]
            log_std = paddle.squeeze(log_std, -1)  #[0: T]
            z = z * paddle.exp(log_std) + mu  #[0: T]

            if i == 0:
                out_mu = mu
                out_log_std = log_std
            else:
                out_mu = out_mu * paddle.exp(log_std) + mu
                out_log_std += log_std

        return z, out_mu, out_log_std


# Gaussian IAF model
class Clarinet(nn.Layer):
    def __init__(self, encoder, teacher, student, stft,
                 min_log_scale=-6.0, lmd=4.0):
        """Clarinet model. Conditional Parallel WaveNet.

        Args:
            encoder (UpsampleNet): an UpsampleNet to upsample mel spectrogram.
            teacher (WaveNet): a WaveNet, the teacher.
            student (ParallelWaveNet): a ParallelWaveNet model, the student.
            stft (STFT): a STFT model to perform differentiable stft transform.
            min_log_scale (float, optional): used only for computing loss, the minimal value of log standard deviation of the output distribution of both the teacher and the student . Defaults to -6.0.
            lmd (float, optional): weight for stft loss. Defaults to 4.0.
        """
        super(Clarinet, self).__init__()
        self.encoder = encoder
        self.teacher = teacher
        self.student = student
        self.stft = stft

        self.lmd = lmd
        self.min_log_scale = min_log_scale

    def forward(self, audio, mel, audio_start, clip_kl=True):
        """Compute loss of Clarinet model.

        Args:
            audio (Variable): shape(B, T_audio), dtype flaot32, ground truth waveform.
            mel (Variable): shape(B, F, T_mel), dtype flaot32, condition(mel spectrogram here).
            audio_start (Variable): shape(B, ), dtype int64, audio starts positions.
            clip_kl (bool, optional): whether to clip kl_loss by maximum=100. Defaults to True.

        Returns:
            Dict(str, Variable)
            loss (Variable): shape(1, ), dtype flaot32, total loss.
            kl (Variable): shape(1, ), dtype flaot32, kl divergence between the teacher's output distribution and student's output distribution.
            regularization (Variable): shape(1, ), dtype flaot32, a regularization term of the KL divergence.
            spectrogram_frame_loss (Variable): shape(1, ), dytpe: float, stft loss, the L1-distance of the magnitudes of the spectrograms of the ground truth waveform and synthesized waveform.
        """
        batch_size, audio_length = audio.shape  # audio clip's length

        z = paddle.randn(audio.shape)
        condition = self.encoder(mel)  # (B, C, T)
        condition_slice = crop(condition, audio_start, audio_length)

        x, s_means, s_scales = self.student(z, condition_slice)  # all [0: T]
        s_means = s_means[:, 1:]  # (B, T-1), time steps [1: T]
        s_scales = s_scales[:, 1:]  # (B, T-1), time steps [1: T]
        s_clipped_scales = paddle.clip(s_scales, self.min_log_scale, 100.)

        # teacher outputs single gaussian
        y = self.teacher(x[:, :-1], condition_slice[:, :, 1:])
        _, t_means, t_scales = paddle.chunk(y, 3, axis=-1)  # time steps [1: T]
        t_means = paddle.squeeze(t_means, [-1])  # (B, T-1), time steps [1: T]
        t_scales = paddle.squeeze(t_scales, [-1])  # (B, T-1), time steps [1: T]
        t_clipped_scales = paddle.clip(t_scales, self.min_log_scale, 100.)

        s_distribution = D.Normal(s_means, paddle.exp(s_clipped_scales))
        t_distribution = D.Normal(t_means, paddle.exp(t_clipped_scales))

        # kl divergence loss, so we only need to sample once? no MC
        kl = s_distribution.kl_divergence(t_distribution)
        if clip_kl:
            kl = paddle.clip(kl, -100., 10.)
        # context size dropped
        kl = paddle.reduce_mean(kl[:, self.teacher.context_size:])
        # major diff here
        regularization = F.mse_loss(t_scales[:, self.teacher.context_size:],
                                    s_scales[:, self.teacher.context_size:])

        # introduce information from real target
        spectrogram_frame_loss = F.mse_loss(
            self.stft.magnitude(audio), self.stft.magnitude(x))
        loss = kl + self.lmd * regularization + spectrogram_frame_loss
        loss_dict = {
            "loss": loss,
            "kl_divergence": kl,
            "regularization": regularization,
            "stft_loss": spectrogram_frame_loss
        }
        return loss_dict

    @paddle.no_grad()
    def synthesis(self, mel):
        """Synthesize waveform using the encoder and the student network.

        Args:
            mel (Variable): shape(B, F, T_mel), the condition(mel spectrogram here).

        Returns:
            Variable: shape(B, T_audio), the synthesized waveform. (T_audio = T_mel * upscale_factor, where upscale_factor is the `upscale_factor` of the encoder.)
        """
        condition = self.encoder(mel)
        samples_shape = (condition.shape[0], condition.shape[-1])
        z = paddle.randn(samples_shape)
        x, s_means, s_scales = self.student(z, condition)
        return x
    

# TODO(chenfeiyu): ClariNetLoss