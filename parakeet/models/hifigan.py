import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.utils import weight_norm, remove_weight_norm
from parakeet.modules.spectral_norm_hook import spectral_norm, remove_spectral_norm
from paddle.nn import SpectralNorm
from itertools import chain


class ResidualBlock1(nn.Layer):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        """number of dilations defines the number of layers of convolutions
        in the residual block"""
        super().__init__()
        # convolutions with dilation
        self.convs1 = nn.LayerList([
            weight_norm(
                nn.Conv1D(channels,
                          channels,
                          kernel_size,
                          dilation=dilations[0],
                          padding="same")),
            weight_norm(
                nn.Conv1D(channels,
                          channels,
                          kernel_size,
                          dilation=dilations[1],
                          padding="same")),
            weight_norm(
                nn.Conv1D(channels,
                          channels,
                          kernel_size,
                          dilation=dilations[2],
                          padding="same"))
        ])
        # convolutions without dilation
        self.convs2 = nn.LayerList([
            weight_norm(
                nn.Conv1D(channels, channels, kernel_size, padding="same")),
            weight_norm(
                nn.Conv1D(channels, channels, kernel_size, padding="same")),
            weight_norm(
                nn.Conv1D(channels, channels, kernel_size, padding="same")),
        ])
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = x2(xt)
            x = x + xt
        return x
    
    def remove_weight_norm(self):
        for layer in chain(self.convs1, self.convs2):
            remove_weight_norm(layer)


class ResidualBlock2(nn.Layer):
    """A simpler alternative to ResidualBlock1."""
    def __init__(self, channels, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.convs = nn.LayerList([
            weight_norm(nn.Conv1D(channels, channels, kernel_size, dilation=dilations[0], padding='same')),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, dilation=dilations[1], padding='same')),
            ])
        
    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = x + xt
        return x
    
    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)
            

class WavGenerator(nn.Layer):
    def __init__(self, 
                 input_size: int, 
                 upsample_init_channels: int, 
                 upsample_rates: List[int], 
                 resblock_kernel_sizes: List[int], 
                 resblock_dilation_sizes: List[List[int]],
                 resblock_type="1"):
        super().__init__()
        self.num_upsample_layers = len(upsample_rates)
        self.num_resblocks_per_upsample = len(resblock_kernel_sizes)
        assert len(resblock_dilation_sizes) == self.num_resblocks_per_upsample
        
        self.conv_pre = weight_norm(nn.Conv1D(input_size, upsample_init_channels, 7, padding="same"))
        
        resblock = ResidualBlock1 if resblock_type == "1" else ResidualBlock2
        
        # Upsampling convtranspose
        self.ups = nn.LayerList()
        for i, u in enumerate(upsample_rates):
            self.ups.append(
                weight_norm(nn.Conv1DTranspose(upsample_init_channels // (2**i), 
                                               upsample_init_channels // (2**(i+1)),
                                               2*u,
                                               u,
                                               padding=u // 2)))
        # Multi-Receptive Field Fusion (MRF)
        self.resblocks = nn.LayerList()
        for i in range(num_upsample_layers):
            ch = upsample_init_channels // (2**(i+1))
            for j, (kernel_size, dilation_sizes) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, kernel_size, dilation_sizes))
                
        self.conv_post = weight_norm(nn.Conv1D(ch, 1, 7, padding="same"))
    
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsample_layers):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_resblocks_per_upsample):
                if xs is None:
                    xs = self.resblocks[i*self.num_upsample_layers + j](x)
                else:
                    xs += self.resblocks[i*self.num_upsample_layers + j](x)
            x = xs / self.num_resblocks_per_upsample
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = paddle.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
            
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        
        
class DiscriminatorP(nn.Layer):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_fn = spectral_norm if use_spectral_norm else weight_norm 
        self.convs = nn.LayerList([
            norm_fn(nn.Conv2D(1, 32, [kernel_size, 1], [stride, 1], padding=[(kernel_size -1) // 2, 0])),
            norm_fn(nn.Conv2D(32, 128, [kernel_size, 1], [stride, 1], padding=[(kernel_size -1) // 2, 0])),
            norm_fn(nn.Conv2D(128, 512, [kernel_size, 1], [stride, 1], padding=[(kernel_size -1) // 2, 0])),
            norm_fn(nn.Conv2D(512, 1024, [kernel_size, 1], [stride, 1], padding=[(kernel_size -1) // 2, 0])),
            norm_fn(nn.Conv2D(1024, 1024, [kernel_size, 1], [stride, 1], padding=[(kernel_size -1) // 2, 0])),
            ])
        self.conv_post = weight_norm(nn.Conv2D(1024, 1, kernel_size=[3, 1], padding=[1, 0]))
    
    def forward(self, x):
        # (B, 1, T) -> (B, T), [multiple (B, C, T/p, p)] time scale shrinks
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, [0, n_pad], mode='reflect', data_format="NCL")
            t += n_pad
        x = paddle.reshape(x, [b, c, t // self.period, self.period])
        
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x, start_axis=1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Layer):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.LayerList([
            DiscriminatorP(p) for p in periods])
    
    def forward(self, y, y_hat):
        # reference, generated
        y_d_rs= []
        y_d_gs =[]
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    

class DiscriminatorS(nn.Layer):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_fn = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.LayerList([
            norm_fn(nn.Conv1D(1, 128, 15, 1, padding='same')),
            norm_fn(nn.Conv1D(128, 128, 41, 2, groups=4, padding='same')),
            norm_fn(nn.Conv1D(128, 256, 41, 2, groups=16, padding='same')),
            norm_fn(nn.Conv1D(256, 512, 41, 4, groups=16, padding='same')),
            norm_fn(nn.Conv1D(512, 1024, 41, 4, groups=16, padding='same')),
            norm_fn(nn.Conv1D(1024, 1024, 41, 1, groups=16, padding='same')),
            norm_fn(nn.Conv1D(1024, 1024, 5, 1, padding='same'))])
        self.conv_post = weight_norm(nn.Conv1D(1024, 1, 3, 1, padding='same'))
    
    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        
        x = paddle.flatten(x, start_axis=1)
        return x, fmap
    

class MultiScaleDiscriminator(nn.Layer):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.LayerList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.LayerList([
            nn.AvgPool1D(4, 2, padding=2),
            nn.AvgPool1D(4, 2, padding=2),
        ])
    
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs            
            

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += paddle.mean(paddle.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = paddle.mean((1-dr)**2)
        g_loss = paddle.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = paddle.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
    
        
        
    
    
    
    