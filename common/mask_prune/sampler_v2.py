#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:46:56 2019

@author: jiahuei
"""
import tensorflow as tf


def bernoulli_sample_sigmoid(logits, sampling_temperature=None, rectify_gradients=False):
    """
    Performs stochastic Bernoulli sampling.
    Accepts raw logits instead of normalised probabilities.
    """
    if sampling_temperature:
        logits = tf.truediv(logits, sampling_temperature)
    probs = tf.nn.sigmoid(logits)
    if rectify_gradients:
        return _bernoulli_sample_rectified(probs)
    else:
        return _bernoulli_sample(probs)


def binarise_sigmoid(logits, threshold=0.5, rectify_gradients=False):
    """
    Performs deterministic binarisation with adjustable threshold.
    Accepts raw logits instead of normalised probabilities.
    """
    probs_fn = tf.nn.sigmoid
    if threshold <= 0 or threshold >= 1:
        raise ValueError('`threshold` must in range (0, 1).')
    if threshold == 0.5:
        if rectify_gradients:
            return _rounding_rectified(probs_fn(logits))
        else:
            return _rounding(probs_fn(logits))
    else:
        probs = tf.subtract(probs_fn(logits), threshold)
        if rectify_gradients:
            return _ceil_rectified(probs)
        else:
            return _ceil(probs)


@tf.custom_gradient
def _bernoulli_sample(probs):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values
    in [0, 1], using the straight-through estimator for the gradient.
    E.g.,:
    if probs is 0.6, bernoulli_sample_sigmoid(logits) will be 1.0 with probability 0.6,
    and 0.0 otherwise, and the gradient will be pass-through (identity).
    
    Inspired by:
    https://gist.github.com/charliememory/035a1607682f6f3663c101d0fe013d9a
    """
    with tf.name_scope('BernoulliSample'):
        rand = tf.random_uniform(tf.shape(probs), minval=0., maxval=1., dtype=probs.dtype)
        sample = tf.ceil(tf.subtract(probs, rand))
    return sample, lambda dy: dy


@tf.custom_gradient
def _bernoulli_sample_rectified(probs):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values
    in [0, 1], using the straight-through estimator for the gradient.
    E.g.,:
    if probs is 0.6, bernoulli_sample_sigmoid(logits) will be 1.0 with probability 0.6,
    and 0.0 otherwise, and the gradient will be pass-through (identity).

    Inspired by:
    https://gist.github.com/charliememory/035a1607682f6f3663c101d0fe013d9a
    """
    raise NotImplementedError
    with tf.name_scope('BernoulliSample'):
        rand = tf.random_uniform(tf.shape(probs), minval=0., maxval=1., dtype=probs.dtype)
        sample = tf.ceil(tf.subtract(probs, rand))
    return sample, lambda dy: tf.multiply(dy, sample)


@tf.custom_gradient
def _rounding(x):
    """
    Rounding with straight-through gradient estimator.
    :param x: Tensor
    :return: Rounded tensor.
    """
    sample = tf.round(x)
    return sample, lambda dy: dy


@tf.custom_gradient
def _rounding_rectified(x):
    """
    Rounding with straight-through gradient estimator.
    :param x: Tensor
    :return: Rounded tensor.
    """
    raise NotImplementedError
    sample = tf.round(x)
    return sample, lambda dy: tf.multiply(dy, sample)


@tf.custom_gradient
def _ceil(x):
    """
    Ceiling with straight-through gradient estimator.
    :param x: Tensor
    :return: Rounded tensor.
    """
    sample = tf.ceil(x)
    return sample, lambda dy: dy


@tf.custom_gradient
def _ceil_rectified(x, rectify_gradients=False):
    """
    Ceiling with straight-through gradient estimator.
    :param x: Tensor
    :return: Rounded tensor.
    """
    raise NotImplementedError
    sample = tf.ceil(x)
    return sample, lambda dy: tf.multiply(dy, sample)


def bernoulli_sample_test():
    # import os
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(4896)
        logits = tf.random_uniform(
            shape=[128, 256],
            minval=-5,
            maxval=5,
            dtype=tf.float32, )
        probs = tf.nn.sigmoid(logits)
        sample = bernoulli_sample_sigmoid(logits)
        grad_probs = tf.gradients(probs, logits)
        grad_sample = tf.gradients(sample, logits)
        with tf.control_dependencies([tf.assert_equal(grad_probs, grad_sample)]):
            grad_probs = tf.identity(grad_probs)
        with tf.control_dependencies([tf.assert_non_negative(sample)]):
            grad_probs = tf.identity(grad_probs)
        init_fn = tf.global_variables_initializer()
    
    sess = tf.Session(graph=g)
    with sess:
        sess.run(init_fn)
        grad_probs = sess.run(grad_probs)
    print('Tests passed: `bernoulli_sample_sigmoid`')

# bernoulli_sample_test()
