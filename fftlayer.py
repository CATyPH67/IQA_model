import math
import keras
import numpy as np
import tensorflow as tf
import scipy.fftpack as fp

@keras.saving.register_keras_serializable(package="fftlayer", name="fftGradLayer")
class fftGradLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rescale = keras.layers.Rescaling(scale=0.5*math.pi, offset=-1)

    def call(self, inputs):
        if tf.is_symbolic_tensor(inputs):
            return self.rescale(inputs)
        gray = tf.image.rgb_to_grayscale(inputs / 255) # normalizing pixel values to [0,1]
        # Compute gradients of batch images
        gxy = tf.image.sobel_edges(gray)             # first order image gradients by y,x axes
        gy = gxy[:,:,:,:,0]
        gx = gxy[:,:,:,:,1]
        g = tf.math.sqrt(gx**2 + gy**2)  # gradient magnitude
        x = tf.concat([gray, gy, gx], axis=3)      # multichannel batch with gray images and gradients in channels
        z = fp.fft(fp.fft(np.array(x), axis=0), axis=1)          # 2d fourier transform
        z = tf.math.log(1 + tf.abs(z))           # normalization of spectrum magnitude
        z = tf.clip_by_value(z, clip_value_min=0, clip_value_max=math.pi)
        return self.rescale(z)

    def get_config(self):
        config = { "batch_shape" : self.batch_shape }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)