# -*- coding: utf-8 -*-
"""
Created on 20 Jun 2019 00:46:44

@author: jiahuei
"""
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import common.ops_v1 as my_ops

_shape = my_ops.shape


class SparseDense(Dense):
    """
    Implements fully connected layer with masked kernel and optionally masked biases.
    """
    
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        print('sparse layer goooooooooo')
        super(SparseDense, self).__init__(units=units,
                                          activation=activation,
                                          use_bias=use_bias,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          bias_constraint=bias_constraint,
                                          trainable=trainable,
                                          name=name,
                                          **kwargs)
        self._input_is_sparse = False
    
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # return super(SparseDense, self).call(inputs)
            inputs = tf.squeeze(inputs)
        outputs = tf.matmul(a=inputs, b=self.kernel,
                            a_is_sparse=self._input_is_sparse, b_is_sparse=True)
        if len(shape) > 2:
            outputs = tf.expand_dims(outputs, 0)
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
    # def call(self, inputs):
    #     inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    #     shape = inputs.get_shape().as_list()
    #     if len(shape) > 2:
    #         # Broadcasting is required for the inputs.
    #         outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
    #                                                                [0]])
    #         # Reshape the output back to the original ndim of the input.
    #         if not context.executing_eagerly():
    #             output_shape = shape[:-1] + [self.units]
    #             outputs.set_shape(output_shape)
    #     else:
    #         outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    #     if self.use_bias:
    #         outputs = nn.bias_add(outputs, self.bias)
    #     if self.activation is not None:
    #         return self.activation(outputs)  # pylint: disable=not-callable
    #     return outputs
