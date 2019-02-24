import numpy as np
import tensorflow as tf
'''
  'artistic style transfer for videos' loss functions
'''


def temporal_loss(x, w, c):
    c = c[np.newaxis, :, :, :]
    D = float(x.size)
    loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss

