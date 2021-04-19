# -*- coding: utf-8 -*-
"""
Created on 13 Jul 2019 18:25:15

@author: jiahuei
"""

import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
_NBINS = 256


def _histogram(values, value_range, nbins=100, dtype=tf.int32, name=None):
    """Return histogram of values.

    Given the tensor `values`, this operation returns a rank 1 histogram counting
    the number of entries in `values` that fell into every bin.  The bins are
    equal width and determined by the arguments `value_range` and `nbins`.

    Args:
      values:  Numeric `Tensor`.
      value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
        values <= value_range[0] will be mapped to hist[0],
        values >= value_range[1] will be mapped to hist[-1].
      nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
      dtype:  dtype for returned histogram.
      name:  A name for this operation (defaults to 'histogram').

    Returns:
      A 1-D `Tensor` holding histogram of values.

    """
    with tf.name_scope(name, 'histogram', [values, value_range, nbins]) as scope:
        values = tf.convert_to_tensor(values, name='values')
        values = tf.reshape(values, [-1])
        value_range = tf.convert_to_tensor(value_range, name='value_range')
        nbins_float = np.float32(nbins)
        
        # Map tensor values that fall within value_range to [0, 1].
        scaled_values = tf.truediv(
            values - value_range[0],
            value_range[1] - value_range[0],
            name='scaled_values')
        
        # map tensor values within the open interval value_range to {0,.., nbins-1},
        # values outside the open interval will be zero or less, or nbins or more.
        indices = tf.floor(nbins_float * scaled_values, name='indices')
        
        # Clip edge cases (e.g. value = value_range[1]) or "outliers."
        indices = tf.cast(
            tf.clip_by_value(indices, 0, nbins_float - 1), tf.int32)
        
        return tf.unsorted_segment_sum(
            tf.ones_like(indices, dtype=dtype), indices, nbins, name=scope)


def compute_cdf_from_histogram(values, value_range, **kwargs):
    """Returns the normalized cumulative distribution of the given values tensor.

    Computes the histogram and uses tf.cumsum to arrive at cdf

    Args:
      values:  Numeric `Tensor`.
      value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      **kwargs: keyword arguments: nbins, name

    Returns:
      A 1-D `Tensor` holding normalized cdf of values.

    """
    nbins = kwargs.get('nbins', _NBINS)
    name = kwargs.get('name', None)
    with tf.name_scope(name, 'cdf', [values, value_range, nbins]):
        histogram = _histogram(
            values, value_range, dtype=tf.float32, nbins=nbins)
        cdf = tf.cumsum(histogram)
        return tf.div(cdf, tf.reduce_max(cdf))


######

val = np.array([[0, 1, 2], [9, 5, 2]]).astype(np.float32)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(4896)
    
    v = tf.identity(val)
    h = compute_cdf_from_histogram(v, [0., 10.], nbins=5)
    init_fn = tf.global_variables_initializer()

sess = tf.Session(graph=g)

with sess:
    # Restore model from checkpoint if provided
    sess.run(init_fn)
    h = sess.run(h)


