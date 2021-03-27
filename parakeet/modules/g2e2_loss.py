import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I


class GE2ELoss(nn.Layer):
    def __init__(self, init_w=10., init_b=-5.):
        super().__init__()
        self.w = self.create_parameter([1], attr=I.Constant(init_w))
        self.b = self.create_parameter([1], attr=I.Constant(init_b))

    def forward(self, embeds):
        # embeds [N, M, C]
        # N - speakers_per_batch,
        # M - utterances_per_speaker
        # C - embed_dim
        sim_matrix = self._build_sim_matrix(embeds)
        _, M, N = sim_matrix.shape
        target = paddle.arange(0, N, dtype="int64").unsqueeze(-1)
        target = paddle.expand(target, [N, M])
        target = paddle.reshape(target, [-1])  # [NM]

        criterion = nn.CrossEntropyLoss()
        loss = criterion(sim_matrix.reshape([-1, N]), target)
        return loss

    def _build_sim_matrix(self, embeds):
        N, M, C = embeds.shape

        # Inclusive centroids (1 per speaker). [N, C]
        centroids_incl = paddle.mean(embeds, axis=1)
        centroids_incl_norm = paddle.norm(centroids_incl,
                                          p=2,
                                          axis=1,
                                          keepdim=True)
        normalized_centroids_incl = centroids_incl / centroids_incl_norm

        # Exclusive centroids (1 per utterance) [N, M, C]
        centroids_excl = paddle.broadcast_to(
            paddle.sum(embeds, axis=1, keepdim=True), embeds.shape) - embeds
        centroids_excl /= (M - 1)
        centroids_excl_norm = paddle.norm(centroids_excl,
                                          p=2,
                                          axis=2,
                                          keepdim=True)
        normalized_centroids_excl = centroids_excl / centroids_excl_norm

        # inter-speaker similarity, NM embeds ~ N centroids
        # [NM, N]
        p1 = paddle.matmul(embeds.reshape([-1, C]),
                           normalized_centroids_incl,
                           transpose_y=True)
        p1 = p1.reshape([-1])  # [NMN]

        # intra-similarity, NM embeds, 1 centroid per embed
        p2 = paddle.bmm(embeds.reshape([-1, 1, C]),
                        normalized_centroids_excl.reshape([-1, C,
                                                           1]))  # (NM, 1, 1)
        p2 = p2.reshape([-1])  # [NM]

        with paddle.no_grad():
            index = paddle.arange(0, N * M, dtype="int64").reshape([N, M])
            index = index * N + paddle.arange(0, N,
                                              dtype="int64").unsqueeze(-1)
            index = paddle.reshape(index, [-1])
        # begin: alternative implementation for scatter
        ones = paddle.ones([N * M * N])
        zeros = paddle.zeros_like(index, dtype=ones.dtype)
        mask_p1 = paddle.scatter(ones, index, zeros)
        p = p1 * mask_p1 + (1 - mask_p1) * paddle.scatter(ones, index, p2)
        # end: alternative implementation for scatter
        # p = paddle.scatter(p1, index, p2) there is a backward bug in scatter

        p = p * self.w + self.b
        p = p.reshape([N, M, N])
        return p
