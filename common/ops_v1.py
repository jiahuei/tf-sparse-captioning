#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:22:42 2017

@author: jiahuei
"""
import os
import logging
import functools
import argparse
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import training
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.slim.python.slim.learning import multiply_gradients, clip_gradient_norms

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import gen_nn_ops, nn_ops, array_ops, rnn_cell_impl
# from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
# from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
# from tensorflow.python.layers.core import Dense
# from packaging import version

slim = tf.contrib.slim

logger = logging.getLogger(__name__)
_DEBUG = False


def dprint(string, is_debug):
    if is_debug:
        print('\n-- DEBUG: {}'.format(string))


def _dprint(string):
    return dprint(string, _DEBUG)


###
# ArgParse functions
###

class ChoiceList(object):
    """
    A Type for ArgParse for validation of choices.
    https://mail.python.org/pipermail/tutor/2011-April/082825.html
    """
    
    def __init__(self, choices):
        self.choices = choices
    
    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.choices)
    
    def __call__(self, csv):
        try:
            args = csv.split(',')
            remainder = sorted(set(args) - set(self.choices))
            if remainder:
                raise ValueError('Invalid choices: {} (choose from {})'.format(remainder, self.choices))
            return args
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)


def convert_none(input_string):
    assert isinstance(input_string, str)
    if input_string.lower() == 'none':
        return None
    else:
        return input_string


def convert_int_csv(input_string):
    assert isinstance(input_string, str)
    try:
        return [int(_) for _ in input_string.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def convert_float_csv(input_string):
    assert isinstance(input_string, str)
    try:
        return [float(_) for _ in input_string.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def convert_csv_str_list(input_string):
    if input_string is None or input_string == '':
        return None
    else:
        assert isinstance(input_string, str)
        try:
            return [_.strip() for _ in input_string.split(',')]
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)


###

def lazy_property(function):
    """
    A decorator for functions. The wrapped function will only be executed once.
    Subsequent calls to it will directly return the result.
    # https://danijar.github.io/structuring-your-tensorflow-models
    """
    attribute = '_cache_' + function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    
    return decorator


def grouper(iterable, group_n, fill_value=0):
    args = [iter(iterable)] * group_n
    return list(itertools.zip_longest(fillvalue=fill_value, *args))


def number_to_base(n, base):
    """Function to convert any base-10 integer to base-N."""
    if base < 2:
        raise ValueError('Base cannot be less than 2.')
    if n < 0:
        sign = -1
        n *= sign
    elif n == 0:
        return [0]
    else:
        sign = 1
    digits = []
    while n:
        digits.append(sign * int(n % base))
        n //= base
    return digits[::-1]


def map_nlist(nlist, fn):
    """
    Maps `fn` to elements in nested list that has arbitrary depth.
    https://stackoverflow.com/a/26133679
    :param nlist: Input list
    :param fn: Function to be applied
    :return: Modified list
    """
    new_list = []
    for i in range(len(nlist)):
        if isinstance(nlist[i], list):
            new_list.append(map_nlist(nlist[i], fn))
        else:
            new_list.append(fn(nlist[i]))
    return new_list


def shape(tensor, replace_None=False):
    """
    Returns the shape of the Tensor as list.
    Can also replace unknown dimension value from `None` to `-1`.
    """
    s = tensor.get_shape().as_list()
    if replace_None:
        s = [-1 if i is None else i for i in s]
    return s


def add_value_summary(data_dict, summary_writer, global_step):
    """Helper to write value to summary."""
    for name, value in data_dict.items():
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step)


def get_model_size(scope_or_list=None, log_path=None):
    if isinstance(scope_or_list, list):
        var = scope_or_list
    else:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_or_list)
    var_shape = []
    for v in var:
        try:
            var_shape.append(shape(v))
        except:
            logger.info('Model size calculation: Encountered opaque variable: ' + v.op.name)
            var_shape.append([])
    model_size = sum([np.prod(v) for v in var_shape])
    if isinstance(scope_or_list, list) or scope_or_list is None:
        name = 'List provided'
    else:
        name = 'Scope `{}`'.format(scope_or_list)
    mssg = "INFO: {} contains {:,d} trainable parameters.".format(
        name, int(model_size))
    print('\n{}\n'.format(mssg))
    mssg = '\r\n{}\r\n\r\n'.format(mssg)
    for i, v in enumerate(var):
        mssg += '{}\r\n{}\r\n\r\n'.format(v.op.name, var_shape[i])
    mssg += '\r\n\r\n'
    if log_path is not None:
        with open(os.path.join(log_path, 'model_size.txt'), 'a') as f:
            f.write(mssg)
    return model_size


###############################################################################


def l2_regulariser(tensor,
                   weight_decay_factor):
    """A `Tensor` -> `Tensor` function that applies L2 weight loss."""
    weight_decay = tf.multiply(tf.nn.l2_loss(tensor),
                               weight_decay_factor,
                               name="L2_weight_loss")
    return weight_decay


def relu(tensor,
         relu_leak_factor):
    """Helper to perform leaky / regular ReLU operation."""
    with tf.name_scope("Leaky_Relu"):
        return tf.maximum(tensor, tensor * relu_leak_factor)


def linear(scope,
           inputs,
           output_dim,
           bias_init=0.0,
           activation_fn=None):
    """
    Helper to perform linear map with optional activation.

    Args:
        scope: name or scope.
        inputs: A 2-D tensor.
        output_dim: The output dimension (second dimension).
        bias_init: Initial value of the bias variable. Pass in `None` for
            linear projection without bias.
        activation_fn: Activation function to be used. (Optional)

    Returns:
        A tensor of shape [inputs_dim[0], `output_dim`].
    """
    with tf.variable_scope(scope):
        input_dim = shape(inputs)[1]
        weight = tf.get_variable(name="weight",
                                 shape=[input_dim, output_dim],
                                 dtype=tf.float32,
                                 initializer=None,
                                 trainable=True)
        if bias_init is None:
            output = tf.matmul(inputs, weight)
        else:
            bias = tf.get_variable(
                name="bias",
                shape=output_dim,
                dtype=tf.float32,
                initializer=tf.constant_initializer(bias_init),
                trainable=True)
            output = tf.matmul(inputs, weight) + bias
        if activation_fn is not None:
            output = activation_fn(output)
        return output


def layer_norm_activate(scope,
                        inputs,
                        activation_fn=None,
                        begin_norm_axis=1):
    """
    Performs Layer Normalization followed by `activation_fn`.

    Args:
        scope: name or scope.
        inputs: A N-D tensor.
        activation_fn: Activation function to be used. (Optional)

    Returns:
        A tensor of the same shape as `inputs`.
    """
    ln_kwargs = dict(
        center=True,
        scale=True,
        activation_fn=activation_fn,
        reuse=False,
        trainable=True,
        scope=scope)
    # if version.parse(tf.__version__) >= version.parse('1.9'):
    try:
        outputs = tf.contrib.layers.layer_norm(
            inputs=inputs,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=-1,
            **ln_kwargs)
    except TypeError:
        print('WARNING: `layer_norm_activate`: `begin_norm_axis` is ignored.')
        outputs = tf.contrib.layers.layer_norm(
            inputs=inputs,
            **ln_kwargs)
    return outputs


def batch_norm_activate(scope,
                        inputs,
                        is_training,
                        activation_fn=None,
                        data_format='NHWC'):
    """
    Performs Batch Normalization followed by `activation_fn`.

    Args:
        scope: name or scope.
        inputs: A N-D tensor.
        activation_fn: Activation function to be used. (Optional)

    Returns:
        A tensor of the same shape as `inputs`.
    """
    
    batch_norm_params = dict(
        epsilon=1e-3,
        decay=0.9997,
        trainable=True,
        activation_fn=None,
        fused=True,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        center=True,
        scale=True,
    )
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            outputs = slim.batch_norm(inputs=inputs,
                                      is_training=is_training,
                                      data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs


def compute_gradients(total_loss,
                      update_ops=None,
                      variables_to_train=None,
                      clip_gradient_norm=0,
                      gradient_multipliers=None,
                      summarize_gradients=False,
                      gate_gradients=tf.train.Optimizer.GATE_OP,
                      aggregation_method=None,
                      colocate_gradients_with_ops=False):
    """Creates an `Operation` that evaluates the gradients and returns the loss.
    Args:
      total_loss: A `Tensor` representing the total loss.
      update_ops: An optional list of updates to execute. If `update_ops` is
        `None`, then the update ops are set to the contents of the
        `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
        it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
        a warning will be displayed.
      variables_to_train: an optional list of variables to train. If None, it will
        default to all tf.trainable_variables().
      clip_gradient_norm: If greater than 0 then the gradients would be clipped
        by it.
      gradient_multipliers: A dictionary of either `Variables` or `Variable` op
        names to the coefficient by which the associated gradient should be
        scaled.
      summarize_gradients: Whether or not add summaries for each gradient.
      gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: Whether or not to try colocating the gradients
        with the ops that generated them.
    Returns:
      A `Tensor` that when evaluated, computes the gradients and returns the total
        loss value.
    """
    
    def transform_grads_fn(grads):
        if gradient_multipliers:
            with tf.name_scope('multiply_grads'):
                grads = multiply_gradients(grads, gradient_multipliers)
        
        # Clip gradients.
        if clip_gradient_norm > 0:
            with tf.name_scope('clip_grads'):
                grads = clip_gradient_norms(grads, clip_gradient_norm)
        return grads
    
    # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
    global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    if update_ops is None:
        update_ops = global_update_ops
    else:
        update_ops = set(update_ops)
    if not global_update_ops.issubset(update_ops):
        tf.logging.warning('update_ops in create_train_op does not contain all the '
                           ' update_ops in GraphKeys.UPDATE_OPS')
    
    # Make sure update_ops are computed before total_loss.
    if update_ops:
        with tf.control_dependencies(update_ops):
            barrier = control_flow_ops.no_op(name='update_barrier')
        total_loss = control_flow_ops.with_dependencies([barrier], total_loss)
    
    if variables_to_train is None:
        # Default to tf.trainable_variables()
        variables_to_train = tf.trainable_variables()
    else:
        # Make sure that variables_to_train are in tf.trainable_variables()
        for v in variables_to_train:
            assert v in tf.trainable_variables()
    
    assert variables_to_train
    
    # Create the gradients. Note that apply_gradients adds the gradient
    # computation to the current graph.
    # We take code from `compute_gradient` to compute gradients
    
    grads = tf.gradients(ys=total_loss,
                         xs=variables_to_train,
                         grad_ys=None,
                         gate_gradients=(gate_gradients == tf.train.Optimizer.GATE_OP),
                         aggregation_method=aggregation_method,
                         colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == tf.train.Optimizer.GATE_GRAPH:
        grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, variables_to_train))
    
    grads_and_vars = transform_grads_fn(grads_and_vars)
    
    # Summarize gradients.
    if summarize_gradients:
        with tf.name_scope('summarize_grads'):
            tf.contrib.training.add_gradients_summaries(grads_and_vars)
    return grads_and_vars


# noinspection PyProtectedMember
def create_train_op(total_loss,
                    optimizers,
                    grads_and_vars,
                    global_step=training._USE_GLOBAL_STEP,
                    check_numerics=True):
    """Creates an `Operation` that evaluates the gradients and returns the loss.
    Args:
      total_loss: A `Tensor` representing the total loss.
      optimizers: A list of tf.Optimizer to use for computing the gradients.
      grads_and_vars: A list of (grad, var) lists. Must have length equal to number of optimizers.
      global_step: A `Tensor` representing the global step variable. If left as
        `_USE_GLOBAL_STEP`, then tf.contrib.framework.global_step() is used.
      check_numerics: Whether or not we apply check_numerics.
    Returns:
      A `Tensor` that when evaluated, computes the gradients and returns the total
        loss value.
    """
    assert len(optimizers) == len(grads_and_vars)
    if global_step is training._USE_GLOBAL_STEP:
        global_step = tf.train.get_or_create_global_step()
    
    # Run each of the optimisers
    grad_updates = []
    for i, optimizer in enumerate(optimizers):
        if i > 0:
            global_step = None
        grads_vars = grads_and_vars[i]
        # Check gradients
        optimizer._assert_valid_dtypes(
            [v for g, v in grads_vars if g is not None and v.dtype != dtypes.resource])
        # Create gradient updates.
        grad_update = optimizer.apply_gradients(grads_vars, global_step=global_step)
        grad_updates.append(grad_update)
    
    with tf.name_scope('train_op'):
        # Make sure total_loss is valid.
        if check_numerics:
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan')
        
        # Ensure the train_tensor computes grad_updates.
        # Produces `total_loss` only after executing `grad_updates`
        train_op = control_flow_ops.with_dependencies(grad_updates, total_loss)
    
    # Add the operation used for training to the 'train_op' collection
    train_ops = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)
    if train_op not in train_ops:
        train_ops.append(train_op)
    return train_op
