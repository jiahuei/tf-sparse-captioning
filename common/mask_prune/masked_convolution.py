# -*- coding: utf-8 -*-
"""
Created on 19 Mar 2020 23:41:27

@author: jiahuei

tensorflow.contrib.layers.python.layers.layers
"""
import functools
import six

from common.mask_prune.masked_layer_v4 import generate_masks, MaskedDense
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
# from tensorflow.python.eager import context
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import function
from tensorflow.python.framework import ops
# from tensorflow.python.framework import sparse_tensor
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
# from tensorflow.python.layers import normalization as normalization_layers
# from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import linalg_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
# from tensorflow.python.ops import sparse_ops
# from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables

# from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None,
                conv_dims=None,
                mask_type=None,
                mask_init_value=None,
                mask_bern_sample=None):
    """Adds an N-D convolution followed by an optional batch_norm layer.
  
    It is required that 1 <= N <= 3.
  
    `convolution` creates a variable called `weights`, representing the
    convolutional kernel, that is convolved (actually cross-correlated) with the
    `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
    provided (such as `batch_norm`), it is then applied. Otherwise, if
    `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
    variable would be created and added the activations. Finally, if
    `activation_fn` is not `None`, it is applied to the activations as well.
  
    Performs atrous convolution with input stride/dilation rate equal to `rate`
    if a value > 1 for any dimension of `rate` is specified.  In this case
    `stride` values != 1 are not supported.
  
    Args:
      inputs: A Tensor of rank N+2 of shape
        `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
        not start with "NC" (default), or
        `[batch_size, in_channels] + input_spatial_shape` if data_format starts
        with "NC".
      num_outputs: Integer, the number of output filters.
      kernel_size: A sequence of N positive integers specifying the spatial
        dimensions of the filters.  Can be a single integer to specify the same
        value for all spatial dimensions.
      stride: A sequence of N positive integers specifying the stride at which to
        compute output.  Can be a single integer to specify the same value for all
        spatial dimensions.  Specifying any `stride` value != 1 is incompatible
        with specifying any `rate` value != 1.
      padding: One of `"VALID"` or `"SAME"`.
      data_format: A string or None.  Specifies whether the channel dimension of
        the `input` and output is the last dimension (default, or if `data_format`
        does not start with "NC"), or the second dimension (if `data_format`
        starts with "NC").  For N=1, the valid values are "NWC" (default) and
        "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
        For N=3, the valid values are "NDHWC" (default) and "NCDHW".
      rate: A sequence of N positive integers specifying the dilation rate to use
        for atrous convolution.  Can be a single integer to specify the same
        value for all spatial dimensions.  Specifying any `rate` value != 1 is
        incompatible with specifying any `stride` value != 1.
      activation_fn: Activation function. The default value is a ReLU function.
        Explicitly set it to None to skip it and maintain a linear activation.
      normalizer_fn: Normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: Normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: Collection to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
      conv_dims: Optional convolution dimensionality, when set it would use the
        corresponding convolution (e.g. 2 for Conv 2D, 3 for Conv 3D, ..). When
        leaved to None it would select the convolution dimensionality based on
        the input rank (i.e. Conv ND, with N = input_rank - 2).
  
    Returns:
      A tensor representing the output of the operation.
  
    Raises:
      ValueError: If `data_format` is invalid.
      ValueError: Both 'rate' and `stride` are not uniformly 1.
    """
    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))
    
    layer_variable_getter = _build_variable_getter({
        'bias': 'biases',
        'kernel': 'weights'
    })
    
    with variable_scope.variable_scope(
            scope, 'Conv', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims
        
        if conv_dims is not None and conv_dims + 2 != input_rank:
            raise ValueError('Convolution expects input with rank %d, got %d' %
                             (conv_dims + 2, input_rank))
        if input_rank == 3:
            layer_class = convolutional_layers.Convolution1D
        elif input_rank == 4:
            layer_class = convolutional_layers.Convolution2D
        elif input_rank == 5:
            layer_class = convolutional_layers.Convolution3D
        else:
            raise ValueError('Convolution not supported for input with rank',
                             input_rank)
        
        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')
        layer = layer_class(
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            data_format=df,
            dilation_rate=rate,
            activation=None,
            use_bias=not normalizer_fn and biases_initializer,
            kernel_initializer=weights_initializer,
            bias_initializer=biases_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=biases_regularizer,
            activity_regularizer=None,
            trainable=trainable,
            name=sc.name,
            dtype=inputs.dtype.base_dtype,
            _scope=sc,
            _reuse=reuse)
        # Insert masks for pruning
        layer.build(inputs.get_shape())
        gen_mask_kwargs = dict(mask_bern_sample=mask_bern_sample,
                               mask_type=mask_type,
                               mask_shape=None,
                               mask_init_value=mask_init_value,
                               dtype=inputs.dtype.base_dtype,
                               get_var_fn=None)
        masked_kernel, masked_bias = generate_masks(kernel=layer.kernel, bias=layer.bias, **gen_mask_kwargs)
        layer.kernel_copy = layer.kernel
        layer.bias_copy = layer.bias
        layer.kernel = masked_kernel
        layer.bias = masked_bias
        outputs = layer.apply(inputs)  # Compute
        layer.kernel = layer.kernel_copy
        layer.bias = layer.bias_copy
        
        # Add variables to collections.
        _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
        if layer.use_bias:
            _add_variable_to_collections(layer.bias, variables_collections, 'biases')
        
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def convolution1d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None,
                  mask_type=None,
                  mask_init_value=None,
                  mask_bern_sample=None):
    return convolution(inputs,
                       num_outputs,
                       kernel_size,
                       stride,
                       padding,
                       data_format,
                       rate,
                       activation_fn,
                       normalizer_fn,
                       normalizer_params,
                       weights_initializer,
                       weights_regularizer,
                       biases_initializer,
                       biases_regularizer,
                       reuse,
                       variables_collections,
                       outputs_collections,
                       trainable,
                       scope,
                       conv_dims=1,
                       mask_type=mask_type,
                       mask_init_value=mask_init_value,
                       mask_bern_sample=mask_bern_sample)


convolution1d.__doc__ = convolution.__doc__


@add_arg_scope
def convolution2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None,
                  mask_type=None,
                  mask_init_value=None,
                  mask_bern_sample=None):
    return convolution(inputs,
                       num_outputs,
                       kernel_size,
                       stride,
                       padding,
                       data_format,
                       rate,
                       activation_fn,
                       normalizer_fn,
                       normalizer_params,
                       weights_initializer,
                       weights_regularizer,
                       biases_initializer,
                       biases_regularizer,
                       reuse,
                       variables_collections,
                       outputs_collections,
                       trainable,
                       scope,
                       conv_dims=2,
                       mask_type=mask_type,
                       mask_init_value=mask_init_value,
                       mask_bern_sample=mask_bern_sample)


convolution2d.__doc__ = convolution.__doc__


@add_arg_scope
def convolution3d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None,
                  mask_type=None,
                  mask_init_value=None,
                  mask_bern_sample=None):
    return convolution(inputs,
                       num_outputs,
                       kernel_size,
                       stride,
                       padding,
                       data_format,
                       rate,
                       activation_fn,
                       normalizer_fn,
                       normalizer_params,
                       weights_initializer,
                       weights_regularizer,
                       biases_initializer,
                       biases_regularizer,
                       reuse,
                       variables_collections,
                       outputs_collections,
                       trainable,
                       scope,
                       conv_dims=3,
                       mask_type=mask_type,
                       mask_init_value=mask_init_value,
                       mask_bern_sample=mask_bern_sample)


convolution3d.__doc__ = convolution.__doc__


def _model_variable_getter(getter,
                           name,
                           shape=None,
                           dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           rename=None,
                           use_resource=None,
                           **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return variables.model_variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        collections=collections,
        trainable=trainable,
        caching_device=caching_device,
        partitioner=partitioner,
        custom_getter=getter,
        use_resource=use_resource)


def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""
    
    # VariableScope will nest the getters
    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)
    
    return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(collections_set,
                                                 collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None,
                    mask_type=None,
                    mask_init_value=None,
                    mask_bern_sample=None):
    """Adds a fully connected layer.
  
    `fully_connected` creates a variable called `weights`, representing a fully
    connected weight matrix, which is multiplied by the `inputs` to produce a
    `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
    `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
    None and a `biases_initializer` is provided then a `biases` variable would be
    created and added the hidden units. Finally, if `activation_fn` is not `None`,
    it is applied to the hidden units as well.
  
    Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
    prior to the initial matrix multiply by `weights`.
  
    Args:
      inputs: A tensor of at least rank 2 and static value for the last dimension;
        i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
      num_outputs: Integer or long, the number of output units in the layer.
      activation_fn: Activation function. The default value is a ReLU function.
        Explicitly set it to None to skip it and maintain a linear activation.
      normalizer_fn: Normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: Normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for all the variables or
        a dictionary containing a different list of collections per variable.
      outputs_collections: Collection to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for variable_scope.
  
    Returns:
       The tensor variable representing the result of the series of operations.
  
    Raises:
      ValueError: If x has rank less than 2 or if its last dimension is not set.
    """
    if not isinstance(num_outputs, six.integer_types):
        raise ValueError('num_outputs should be int or long, got %s.' %
                         (num_outputs,))
    
    layer_variable_getter = _build_variable_getter({
        'bias': 'biases',
        'kernel': 'weights'
    })
    
    with variable_scope.variable_scope(
            scope,
            'fully_connected', [inputs],
            reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        if mask_type is not None:
            dense_cls = MaskedDense
            mask_kwargs = dict(mask_type=mask_type,
                               mask_init_value=mask_init_value,
                               mask_bern_sample=mask_bern_sample)
        else:
            dense_cls = core_layers.Dense
            mask_kwargs = {}
        layer = dense_cls(
            units=num_outputs,
            activation=None,
            use_bias=not normalizer_fn and biases_initializer,
            kernel_initializer=weights_initializer,
            bias_initializer=biases_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=biases_regularizer,
            activity_regularizer=None,
            trainable=trainable,
            name=sc.name,
            dtype=inputs.dtype.base_dtype,
            _scope=sc,
            _reuse=reuse,
            **mask_kwargs)
        outputs = layer.apply(inputs)
        
        # Add variables to collections.
        _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
        if layer.bias is not None:
            _add_variable_to_collections(layer.bias, variables_collections, 'biases')
        
        # Apply normalizer function / layer.
        if normalizer_fn is not None:
            if not normalizer_params:
                normalizer_params = {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def separable_convolution2d(
        inputs,
        num_outputs,
        kernel_size,
        depth_multiplier,
        stride=1,
        padding='SAME',
        data_format=DATA_FORMAT_NHWC,
        rate=1,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None,
        mask_type=None,
        mask_init_value=None,
        mask_bern_sample=None):
    """Adds a depth-separable 2D convolution with optional batch_norm layer.
  
    This op first performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. If `num_outputs`
    is not None, it adds a pointwise convolution that mixes channels, creating a
    variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
  
    Args:
      inputs: A tensor of size [batch_size, height, width, channels].
      num_outputs: The number of pointwise convolution output filters. If is
        None, then we skip the pointwise convolution stage.
      kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `num_filters_in * depth_multiplier`.
      stride: A list of length 2: [stride_height, stride_width], specifying the
        depthwise convolution stride. Can be an int if both strides are the same.
      padding: One of 'VALID' or 'SAME'.
      data_format: A string. `NHWC` (default) and `NCHW` are supported.
      rate: A list of length 2: [rate_height, rate_width], specifying the dilation
        rates for atrous convolution. Can be an int if both rates are the same.
        If any value is larger than one, then both stride values need to be one.
      activation_fn: Activation function. The default value is a ReLU function.
        Explicitly set it to None to skip it and maintain a linear activation.
      normalizer_fn: Normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: Normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: Collection to add the outputs.
      trainable: Whether or not the variables should be trainable or not.
      scope: Optional scope for variable_scope.
  
    Returns:
      A `Tensor` representing the output of the operation.
    Raises:
      ValueError: If `data_format` is invalid.
    """
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    layer_variable_getter = _build_variable_getter({
        'bias': 'biases',
        'depthwise_kernel': 'depthwise_weights',
        'pointwise_kernel': 'pointwise_weights'
    })
    
    gen_mask_kwargs = dict(mask_bern_sample=mask_bern_sample,
                           mask_type=mask_type,
                           mask_shape=None,
                           mask_init_value=mask_init_value,
                           dtype=inputs.dtype.base_dtype,
                           get_var_fn=None)
    
    with variable_scope.variable_scope(
            scope,
            'SeparableConv2d', [inputs],
            reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        
        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')
        if num_outputs is not None:
            # Apply separable conv using the SeparableConvolution2D layer.
            layer = convolutional_layers.SeparableConvolution2D(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                data_format=df,
                dilation_rate=utils.two_element_tuple(rate),
                activation=None,
                depth_multiplier=depth_multiplier,
                use_bias=not normalizer_fn and biases_initializer,
                depthwise_initializer=weights_initializer,
                pointwise_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                depthwise_regularizer=weights_regularizer,
                pointwise_regularizer=weights_regularizer,
                bias_regularizer=biases_regularizer,
                activity_regularizer=None,
                trainable=trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=reuse)
            # Insert masks for pruning
            layer.build(inputs.get_shape())
            masked_depthwise_kernel, masked_bias = generate_masks(
                kernel=layer.depthwise_kernel, bias=layer.bias, **gen_mask_kwargs)
            masked_pointwise_kernel, _ = generate_masks(
                kernel=layer.pointwise_kernel, bias=None, **gen_mask_kwargs)
            layer.depthwise_kernel_copy = layer.depthwise_kernel
            layer.pointwise_kernel_copy = layer.pointwise_kernel
            layer.bias_copy = layer.bias
            layer.depthwise_kernel = masked_depthwise_kernel
            layer.pointwise_kernel = masked_pointwise_kernel
            layer.bias = masked_bias
            outputs = layer.apply(inputs)  # Compute
            layer.depthwise_kernel = layer.depthwise_kernel_copy
            layer.pointwise_kernel = layer.pointwise_kernel_copy
            layer.bias = layer.bias_copy
            
            # Add variables to collections.
            _add_variable_to_collections(layer.depthwise_kernel,
                                         variables_collections, 'weights')
            _add_variable_to_collections(layer.pointwise_kernel,
                                         variables_collections, 'weights')
            if layer.bias is not None:
                _add_variable_to_collections(layer.bias, variables_collections,
                                             'biases')
            
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            # Actually apply depthwise conv instead of separable conv.
            dtype = inputs.dtype.base_dtype
            kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
            stride_h, stride_w = utils.two_element_tuple(stride)
            num_filters_in = utils.channel_dimension(
                inputs.get_shape(), df, min_rank=4)
            weights_collections = utils.get_variable_collections(
                variables_collections, 'weights')
            
            depthwise_shape = [kernel_h, kernel_w, num_filters_in, depth_multiplier]
            depthwise_weights = variables.model_variable(
                'depthwise_weights',
                shape=depthwise_shape,
                dtype=dtype,
                initializer=weights_initializer,
                regularizer=weights_regularizer,
                trainable=trainable,
                collections=weights_collections)
            masked_depthwise_weights, _ = generate_masks(
                kernel=depthwise_weights, bias=None, **gen_mask_kwargs)
            
            strides = [1, 1, stride_h,
                       stride_w] if data_format.startswith('NC') else [
                1, stride_h, stride_w, 1
            ]
            
            outputs = nn.depthwise_conv2d(
                inputs,
                masked_depthwise_weights,
                strides,
                padding,
                rate=utils.two_element_tuple(rate),
                data_format=data_format)
            num_outputs = depth_multiplier * num_filters_in
            
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                outputs = normalizer_fn(outputs, **normalizer_params)
            else:
                if biases_initializer is not None:
                    biases_collections = utils.get_variable_collections(
                        variables_collections, 'biases')
                    biases = variables.model_variable(
                        'biases',
                        shape=[
                            num_outputs,
                        ],
                        dtype=dtype,
                        initializer=biases_initializer,
                        regularizer=biases_regularizer,
                        trainable=trainable,
                        collections=biases_collections)
                    # TODO: bias is not masked currently
                    outputs = nn.bias_add(outputs, biases, data_format=data_format)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


# Simple aliases which remove the activation_fn parameter.
elu = functools.partial(fully_connected, activation_fn=nn.elu)
relu = functools.partial(fully_connected, activation_fn=nn.relu)
relu6 = functools.partial(fully_connected, activation_fn=nn.relu6)
linear = functools.partial(fully_connected, activation_fn=None)

# Simple alias.
conv2d = convolution2d
conv3d = convolution3d
# conv2d_in_plane = convolution2d_in_plane
separable_conv2d = separable_convolution2d
