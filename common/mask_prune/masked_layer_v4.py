# -*- coding: utf-8 -*-
"""
Created on 20 Jun 2019 00:46:44

@author: jiahuei
"""
import tensorflow as tf
from tensorflow.contrib.model_pruning.python.layers import core_layers
from tensorflow.python.layers.core import Dense
from common.mask_prune import sampler_v2 as sampler
import common.ops_v1 as my_ops

_shape = my_ops.shape

TRAIN_MASK = 'train_mask'
STRUCTURED = 'structured'
REGULAR = 'regular'
HYBRID = 'hybrid'

MAG_BLIND = 'mag_blind'
MAG_UNIFORM = 'mag_uniform'
MAG_DIST = 'mag_dist'
MAG_GRAD_BLIND = 'mag_grad_blind'
MAG_GRAD_UNIFORM = 'mag_grad_uniform'

SNIP = 'snip'
SRINIVAS = 'srinivas'
LOTTERY = 'lottery'

SUPER_MASKS = [STRUCTURED, REGULAR, SRINIVAS]
MAG_ANNEAL = [MAG_GRAD_BLIND, MAG_GRAD_UNIFORM]
MAG_HARD = [MAG_BLIND, MAG_UNIFORM, MAG_DIST]
MAG_PRUNE_MASKS = MAG_HARD + MAG_ANNEAL + [SNIP]
VALID_MASKS = [TRAIN_MASK] + SUPER_MASKS + MAG_PRUNE_MASKS + [LOTTERY]

MASK_COLLECTION = core_layers.MASK_COLLECTION  # 'masks'
MASKED_WEIGHT_COLLECTION = core_layers.MASKED_WEIGHT_COLLECTION  # 'masked_weights'
WEIGHT_COLLECTION = core_layers.WEIGHT_COLLECTION  # 'kernel'
MASKED_WEIGHT_NAME = core_layers.MASKED_WEIGHT_NAME  # 'weights/masked_weight'


# def multiply_st_mask(mask, kernel, name=None):
#     """
#     Kernel masking op with straight-through backprop for masked kernel.
#     Thus gradients received by kernel is as if it is not masked.
#
#     :param mask: Mask variable, values in [0, 1].
#     :param kernel: Kernel / weight variable.
#     :param name: A name for the operation (optional).
#     :return: Masked kernel, ie mask * kernel.
#     """
#     @tf.custom_gradient
#     def _mul(_m, _k):
#         return tf.multiply(_m, _k, name), lambda dy: (dy * _k, dy)
#     return _mul(mask, kernel)


def generate_masks(kernel,
                   bias,
                   mask_bern_sample,
                   mask_type=REGULAR,
                   mask_shape=None,
                   mask_init_value=None,
                   mask_scope=None,
                   dtype=None,
                   get_var_fn=None):
    assert mask_type in VALID_MASKS
    if mask_init_value is None:
        mask_init_value = 0.
    if dtype is None:
        dtype = kernel.dtype
    if get_var_fn is None:
        get_var_fn = tf.get_variable
    
    if mask_type in SUPER_MASKS:
        if mask_type == SRINIVAS:
            raise NotImplementedError
            sample_fn = tf.identity
        else:
            # Supermasks
            if mask_bern_sample:
                sample_fn = sampler.bernoulli_sample_sigmoid
            else:
                sample_fn = sampler.binarise_sigmoid
        mask_trainable = True
    else:
        # Regular pruning
        mask_init_value = 1.
        mask_trainable = False
        sample_fn = tf.identity
    
    with tf.name_scope('gen_masks'):
        kernel_shape = _shape(kernel)
        
        if mask_type == STRUCTURED:
            if len(kernel_shape) == 1:
                mask_shape = kernel_shape
            elif len(kernel_shape) == 2:
                if mask_shape is None:
                    mask_shape = [1, kernel_shape[-1]]
                else:
                    # This is to accommodate LSTM kernels
                    assert mask_shape[0] == 1
                    assert kernel_shape[1] % mask_shape[1] == 0
            else:
                # Convolutional
                mask_shape = [1] * (len(kernel_shape) - 1) + kernel_shape[-1]
        else:
            mask_shape = kernel_shape
        
        mask_init = tf.initializers.constant(value=mask_init_value, dtype=dtype, verify_shape=False)
        
        # Kernel mask
        name = kernel.op.name.split('/')[-1]
        if mask_scope:
            name = '{}/{}'.format(mask_scope, name)
        kernel_mask = get_var_fn(name='{}/mask'.format(name),
                                 shape=mask_shape,
                                 initializer=mask_init,
                                 trainable=mask_trainable,
                                 dtype=dtype)
        
        # Sample mask
        kernel_mask_sampled = sample_fn(kernel_mask)
        if mask_type == STRUCTURED and len(kernel_shape) == 2:
            kernel_mask_sampled = tf.tile(
                kernel_mask_sampled, multiples=[kernel_shape[0], kernel_shape[1] // mask_shape[-1]])
        assert _shape(kernel_mask_sampled) == kernel_shape
        
        # Mask kernel
        masked_kernel = tf.multiply(kernel_mask_sampled, kernel, MASKED_WEIGHT_NAME)
        
        # Mask bias
        masked_bias = bias
        _mask_bias = mask_type == STRUCTURED and bias is not None
        if _mask_bias:
            raise NotImplementedError
            masked_bias = tf.multiply(kernel_mask_sampled[0, :], bias, 'masked_bias')
    
    # Add to collections
    if kernel_mask not in tf.get_collection_ref(MASK_COLLECTION):
        if mask_type in SUPER_MASKS:
            tf.add_to_collection('masks_sampled', kernel_mask_sampled)
        tf.add_to_collection(MASK_COLLECTION, kernel_mask)
        tf.add_to_collection(MASKED_WEIGHT_COLLECTION, masked_kernel)
        tf.add_to_collection(WEIGHT_COLLECTION, kernel)
        if _mask_bias:
            if mask_type in SUPER_MASKS:
                tf.add_to_collection('masks_sampled', kernel_mask_sampled)
            tf.add_to_collection(MASK_COLLECTION, kernel_mask)
            tf.add_to_collection(MASKED_WEIGHT_COLLECTION, masked_bias)
            tf.add_to_collection(WEIGHT_COLLECTION, bias)
    
    return masked_kernel, masked_bias


class MaskedDense(Dense):
    """
    Implements fully connected layer with masked kernel and optionally masked biases.
    """
    
    def __init__(self,
                 mask_type,
                 mask_init_value,
                 mask_bern_sample,
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
        super(MaskedDense, self).__init__(units=units,
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
        self._mask_type = mask_type
        self._mask_init_value = mask_init_value
        self._mask_bern_sample = mask_bern_sample
    
    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        super(MaskedDense, self).build(input_shape)
        self.built = False
        self._kernel_copy = self.kernel
        self._bias_copy = self.bias
        
        gen_mask_kwargs = dict(mask_bern_sample=self._mask_bern_sample,
                               mask_type=self._mask_type,
                               mask_shape=None,
                               mask_init_value=self._mask_init_value,
                               dtype=self.dtype,
                               get_var_fn=self.add_variable)
        self._masked_kernel, self._masked_bias = generate_masks(kernel=self.kernel,
                                                                bias=self.bias,
                                                                **gen_mask_kwargs)
        self.built = True
    
    # noinspection PyAttributeOutsideInit
    def call(self, inputs):
        # Temporarily swap in masked variables
        self.kernel = self._masked_kernel
        self.bias = self._masked_bias
        
        # Call
        outputs = super(MaskedDense, self).call(inputs)
        
        # Swap back just in case
        self.kernel = self._kernel_copy
        self.bias = self._bias_copy
        return outputs


# noinspection PyAbstractClass
class MaskedBasicLSTMCell(tf.contrib.rnn.LSTMBlockCell):
    """Basic LSTM recurrent network cell with pruning.

    Overrides the call method of tensorflow BasicLSTMCell and injects the weight
    masks.
    """
    
    def __init__(self,
                 mask_type,
                 mask_init_value,
                 mask_bern_sample,
                 num_units,
                 forget_bias=1.0,
                 reuse=None,
                 name='basic_lstm_cell'):
        """Initialize the basic LSTM cell with pruning.

        Args:
            mask_type: `regular`, `structured`, `hybrid`.
            mask_init_value: float, the init value for masks.
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
                Must set to `0.0` manually when restoring from CudnnLSTM-trained
                checkpoints.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.  If not `True`, and the existing scope already has
                the given variables, an error is raised.
            name: String, the name of the layer. Layers with the same name will
                share weights, but to avoid mistakes we require reuse=True in such
                cases.

            When restoring from CudnnLSTM-trained checkpoints, must use
            CudnnCompatibleLSTMCell instead.
        """
        super(MaskedBasicLSTMCell, self).__init__(num_units=num_units,
                                                  forget_bias=forget_bias,
                                                  cell_clip=-1,
                                                  use_peephole=False,
                                                  reuse=reuse,
                                                  name=name)
        self._mask_type = mask_type
        self._mask_init_value = mask_init_value
        self._mask_bern_sample = mask_bern_sample
    
    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        # Call the build method of the parent class.
        super(MaskedBasicLSTMCell, self).build(inputs_shape)
        self._kernel_copy = self._kernel
        self._bias_copy = self._bias
        
        self.built = False
        
        # Create masks, masked kernels and masked biases
        h_depth = self._num_units
        if self._mask_type == STRUCTURED:
            mask_shape = [1, h_depth]
        else:
            mask_shape = None
        gen_mask_kwargs = dict(mask_bern_sample=self._mask_bern_sample,
                               mask_type=self._mask_type,
                               mask_shape=mask_shape,
                               mask_init_value=self._mask_init_value,
                               dtype=self.dtype,
                               get_var_fn=self.add_variable)
        
        self._masked_kernel, self._masked_bias = generate_masks(kernel=self._kernel,
                                                                bias=self._bias,
                                                                **gen_mask_kwargs)
        self.built = True
    
    # noinspection PyAttributeOutsideInit
    def call(self, inputs, state):
        # Temporarily swap in masked variables
        self._kernel = self._masked_kernel
        self._bias = self._masked_bias
        
        # Call
        new_h, new_state = super(MaskedBasicLSTMCell, self).call(inputs, state)
        
        # Swap back just in case
        self._kernel = self._kernel_copy
        self._bias = self._bias_copy
        return new_h, new_state


# noinspection PyAbstractClass
class MaskedGRUCell(tf.contrib.rnn.GRUBlockCellV2):
    def __init__(self,
                 mask_type,
                 mask_init_value,
                 mask_bern_sample,
                 num_units,
                 reuse=None,
                 name='gru_cell'):
        super(MaskedGRUCell, self).__init__(num_units=num_units,
                                            reuse=reuse,
                                            name=name)
        self._mask_type = mask_type
        self._mask_init_value = mask_init_value
        self._mask_bern_sample = mask_bern_sample
    
    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        # Call the build method of the parent class.
        super(MaskedGRUCell, self).build(inputs_shape)
        self._gate_kernel_copy = self._gate_kernel
        self._gate_bias_copy = self._gate_bias
        self._candidate_kernel_copy = self._candidate_kernel
        self._candidate_bias_copy = self._candidate_bias
        
        self.built = False
        
        # Create masks, masked kernels and masked biases
        h_depth = self._cell_size
        if self._mask_type == STRUCTURED:
            # Although only applicable for `gate_kernel`,
            # `candidate_kernel` has shape [n, h_depth] so it does not matter
            mask_shape = [1, h_depth]
        else:
            mask_shape = None
        gen_mask_kwargs = dict(mask_bern_sample=self._mask_bern_sample,
                               mask_type=self._mask_type,
                               mask_shape=mask_shape,
                               mask_init_value=self._mask_init_value,
                               dtype=self.dtype,
                               get_var_fn=self.add_variable)
        
        # with tf.variable_scope('gates'):
        self._masked_gate_kernel, self._masked_gate_bias = generate_masks(
            kernel=self._gate_kernel,
            bias=self._gate_bias,
            mask_scope='gates',
            **gen_mask_kwargs)
        
        # with tf.variable_scope('candidate'):
        self._masked_candidate_kernel, self._masked_candidate_bias = generate_masks(
            kernel=self._candidate_kernel,
            bias=self._candidate_bias,
            mask_scope='candidate',
            **gen_mask_kwargs)
        self.built = True
    
    # noinspection PyAttributeOutsideInit
    def call(self, inputs, state):
        # Temporarily swap in masked variables
        self._gate_kernel = self._masked_gate_kernel
        self._gate_bias = self._masked_gate_bias
        self._candidate_kernel = self._masked_candidate_kernel
        self._candidate_bias = self._masked_candidate_bias
        
        # Call
        new_h, new_h = super(MaskedGRUCell, self).call(inputs, state)
        
        # Swap back just in case
        self._gate_kernel = self._gate_kernel_copy
        self._gate_bias = self._gate_bias_copy
        self._candidate_kernel = self._candidate_kernel_copy
        self._candidate_bias = self._candidate_bias_copy
        return new_h, new_h
