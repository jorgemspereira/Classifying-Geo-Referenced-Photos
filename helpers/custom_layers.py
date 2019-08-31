import tensorflow as tf
from keras.engine import Layer
from keras import backend as K
from keras.utils import get_custom_objects


def softargmax(x, beta=1e10):
    x = tf.convert_to_tensor(x)
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)


class OutputLayer(Layer):

    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'outputLayer'

    def call(self, inputs, **kwargs):
        result = softargmax(inputs[0])
        return K.switch(K.less(result, 1.0),
                        K.zeros_like(inputs[1]),
                        K.switch(K.less(result, 2.0),
                                 inputs[1],
                                 inputs[2]))

    def get_config(self): return super(OutputLayer, self).get_config()

    def compute_output_shape(self, input_shape): return (None, 1)


get_custom_objects().update({'outputlayer': OutputLayer()})
