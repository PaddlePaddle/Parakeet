
"""
This modules contains normalizers for spectrogram magnitude.
Normalizers are invertible transformations. They can be used to process 
magnitude of spectrogram before training and can also be used to recover from 
the generated spectrogram so as to be used with vocoders like griffin lim.

The base class describe the interface. `transform` is used to perform 
transformation and `inverse` is used to perform the inverse transformation.
"""
import numpy as np

class NormalizerBase(object):
    def transform(self, spec):
        raise NotImplementedError("transform must be implemented")
    
    def inverse(self, normalized):
        raise NotImplementedError("inverse must be implemented")

class LogMagnitude(NormalizerBase):
    def __init__(self, min=1e-7):
        self.min = min
    
    def transform(self, x):
        x = np.maximum(x, self.min)
        x = np.log(x)
        return x
    
    def inverse(self, x):
        return np.exp(x)
    
class UnitMagnitude(NormalizerBase):
    # dbscale and (0, 1) normalization
    pass