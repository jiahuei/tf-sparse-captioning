# -*- coding: utf-8 -*-
"""
Created on 12 Jul 2019 22:45:39

@author: jiahuei
"""
import tensorflow as tf
import numpy as np
import os
import logging
from tensorflow.contrib.model_pruning.python import pruning_utils
# from tensorflow.contrib.model_pruning.python import pruning
from common import ops_v1 as ops
from common.mask_prune import masked_layer
from common.mask_prune import sampler

logger = logging.getLogger(__name__)
# _NBINS = 256
_NBINS = 512
LOSS_TYPE = ['L1', 'L2', 'hinge_L1']
pjoin = os.path.join
_shape = ops.shape


def calculate_weight_sparsities(weights, weight_op_names=None):
    return calculate_sparsities(tensor_list=weights,
                                count_nnz_fn=lambda x: tf.count_nonzero(x, axis=None, dtype=tf.float32),
                                tensor_op_names=weight_op_names)


def calculate_mask_sparsities(sampled_masks, mask_op_names):
    return calculate_sparsities(tensor_list=sampled_masks,
                                count_nnz_fn=tf.reduce_sum,
                                tensor_op_names=mask_op_names)


def calculate_sparsities(tensor_list, count_nnz_fn, tensor_op_names=None):
    if tensor_op_names is not None:
        assert isinstance(tensor_op_names, list)
    tensor_sizes = [tf.to_float(tf.reduce_prod(_shape(t))) for t in tensor_list]
    tensor_nnz = []
    tensor_sps = []
    for i, m in enumerate(tensor_list):
        m_nnz = count_nnz_fn(m)
        m_sps = tf.subtract(1.0, tf.divide(m_nnz, tensor_sizes[i]))
        tensor_nnz.append(m_nnz)
        if tensor_op_names is None:
            m_name = ''
        else:
            # m_name = '/'.join(tensor_op_names[i].split('/')[-3:])
            m_name = tensor_op_names[i]
        tensor_sps.append((m_name, m_sps))
        # tf.summary.scalar(m_name, m_sps)
    total_nnz = tf.add_n(tensor_nnz)
    total_size = tf.add_n(tensor_sizes)
    total_sparsity = tf.subtract(1.0, tf.divide(total_nnz, total_size))
    return total_sparsity, total_nnz, tensor_sps


def mask_sparsity_summaries(masks_list, mask_op_names):
    """
    Add summary ops for mask sparsity levels. The masks provided must have binary values (either 0. or 1.).
    :param masks_list:
    :param mask_op_names:
    :return:
    """
    with tf.name_scope('sparsity'):
        total_sparsity, total_nnz, mask_sps = calculate_mask_sparsities(masks_list, mask_op_names)
        for sps in mask_sps:
            tf.summary.scalar(*sps)
    tf.summary.scalar('total_nnz', total_nnz)
    tf.summary.scalar('total_sparsity', total_sparsity)
    return total_sparsity


def write_sparsities_to_file(log_dir, val):
    assert 'global_step' in val
    assert 'total_sparsity' in val
    assert 'total_nnz' in val
    assert 'mask_sps' in val
    
    out = [
        '{}'.format(val['global_step']),
        '{:9.7f}'.format(val['total_sparsity']),
        '{:d}'.format(int(val['total_nnz']))
    ]
    out += ['{:9.7f}'.format(sps[1]) for sps in val['mask_sps']]
    out = '\r\n' + ','.join(out)
    
    fpath = pjoin(log_dir, 'sparsity_values.csv')
    if not os.path.isfile(fpath):
        headers = 'Global step,Total sparsity,Total NNZ,'
        headers += ','.join([str(sps[0]) for sps in val['mask_sps']])
        out = headers + out
    with open(fpath, 'a') as f:
        f.write(out)


def get_masks(sampling_method='binarise_round', exclude_scopes=None):
    masks = tf.contrib.model_pruning.get_masks()
    mask_sampled_ref = tf.get_collection('masks_sampled')
    is_mag_prune = len(mask_sampled_ref) == 0
    if exclude_scopes is not None:
        assert isinstance(exclude_scopes, (list, tuple))
        masks = tf.contrib.framework.filter_variables(
            var_list=masks,
            include_patterns=None,
            exclude_patterns=exclude_scopes,
            reg_search=True)
        mask_sampled_ref = tf.contrib.framework.filter_variables(
            var_list=mask_sampled_ref,
            include_patterns=None,
            exclude_patterns=exclude_scopes,
            reg_search=True)
    if is_mag_prune:
        logger.debug('get_mask(): Should be magnitude pruning')
        return masks, masks
    else:
        assert sampling_method in ['binarise_round', 'rand', 'sigmoid']
        if sampling_method == 'rand':
            sampled_masks = mask_sampled_ref[:]
        else:
            sampled_masks = []
            for m, m_sampled in zip(masks, mask_sampled_ref):
                if sampling_method == 'binarise_round':
                    m = sampler.binarise_sigmoid(m)
                else:
                    raise NotImplementedError
                    m = tf.nn.sigmoid(m)
                m_sampled_s = _shape(m_sampled)
                if _shape(m) != m_sampled_s:
                    # Mask mode is Structured
                    m = tf.tile(m, multiples=[m_sampled_s[0], m_sampled_s[1] // _shape(m)[-1]])
                sampled_masks.append(m)
        return sampled_masks, masks


def get_weights(exclude_scopes=None):
    weights = tf.contrib.model_pruning.get_weights()
    if exclude_scopes is not None:
        assert isinstance(exclude_scopes, (list, tuple))
        weights = tf.contrib.framework.filter_variables(
            var_list=weights,
            include_patterns=None,
            exclude_patterns=exclude_scopes,
            reg_search=True)
    return weights


def get_mask_assign_ops(mask_type, sparsity_target, exclude_scopes, loss=None):
    if mask_type in masked_layer.MAG_PRUNE_MASKS + [masked_layer.LOTTERY]:
        masks, _ = get_masks(exclude_scopes=exclude_scopes)
        weights = get_weights(exclude_scopes=exclude_scopes)
    else:
        raise ValueError('Invalid mask type. Must be one of {}'.format(
            # masked_layer.MAG_PRUNE_MASKS + masked_layer.MASK_PRUNE))
            masked_layer.MAG_PRUNE_MASKS))
    assert len(weights) == len(masks)
    assert len(masks) > 0
    
    with tf.name_scope('mask_assign_ops'):
        if mask_type == masked_layer.SNIP:
            # Maybe accumulate saliency
            with tf.variable_scope('accum_saliency'):
                zero_init = tf.initializers.zeros(loss.dtype)
                var_kwargs = dict(dtype=loss.dtype, initializer=zero_init, trainable=False)
                saliency = [tf.get_variable('saliency_m{}'.format(i), shape=_shape(m), **var_kwargs)
                            for i, m in enumerate(masks)]
                # saliency_batch = [tf.abs(s) for s in tf.gradients(ys=loss, xs=masks)]
                saliency_batch = [s for s in tf.gradients(ys=loss, xs=masks)]
                # Ops for accumulating saliency
                accum_ops = [sal.assign_add(sal_b) for (sal, sal_b) in zip(saliency, saliency_batch)]
            # saliency = [tf.abs(s) for s in tf.gradients(ys=loss, xs=masks)]
            mask_ori_shape = [_shape(m) for m in masks]
            mask_num_elems = [np.prod(m) for m in mask_ori_shape]
            saliency_vec = tf.concat([tf.reshape(s, [-1]) for s in saliency], axis=0)
            saliency_vec = tf.abs(saliency_vec)
            saliency_vec = tf.divide(saliency_vec, tf.reduce_sum(saliency_vec))
            num_params = _shape(saliency_vec)[0]
            kappa = int(round(num_params * (1. - sparsity_target)))
            _, ind = tf.nn.top_k(saliency_vec, k=kappa, sorted=True)
            mask_sparse_vec = tf.sparse_to_dense(ind, tf.shape(saliency_vec),
                                                 tf.ones_like(ind, dtype=tf.float32),
                                                 validate_indices=False)
            mask_sparse_split = tf.split(mask_sparse_vec, mask_num_elems)
            mask_sparse = [tf.reshape(m, ms) for m, ms in zip(mask_sparse_split, mask_ori_shape)]
            assign_ops = [tf.assign(m, new_mask) for m, new_mask in zip(masks, mask_sparse)]
            return assign_ops, accum_ops
        
        elif mask_type == masked_layer.MAG_DIST:
            # Magnitude pruning, class-distribution
            # Calculate standard dev of each class
            # Transform weights as positive factor of standard dev, ie w' = | (w - mean) / std_dev |
            # Reshape and concat all factorised weights, and calculate threshold
            # The rest of the operations are same as class-blind
            abs_weights = []
            for w in weights:
                mean, var = tf.nn.moments(w, axes=list(range(len(_shape(w)))))
                std_dev = tf.sqrt(var)
                w = tf.abs(tf.divide(tf.subtract(w, mean), std_dev))
                abs_weights.append(w)
            criterion = [tf.concat([tf.reshape(w, [-1]) for w in abs_weights], axis=0)]
        
        else:
            abs_weights = [tf.abs(w) for w in weights]
            # if mask_type in masked_layer.MAG_BLIND + masked_layer.MASK_BLIND:
            if mask_type == masked_layer.MAG_UNIFORM:
                # Magnitude pruning, class-uniform
                criterion = abs_weights
            elif mask_type in (masked_layer.MAG_BLIND, masked_layer.LOTTERY):
                # Magnitude pruning, class-blind
                # We reshape all the weights into a vector, and concat them
                criterion = [tf.concat([tf.reshape(w, [-1]) for w in abs_weights], axis=0)]
        
        # len == 1 for class-blind, and len == len(weights) for others
        thresholds = [_get_threshold(c, sparsity_target, nbins=_NBINS) for c in criterion]
        if len(thresholds) != len(masks):
            assert len(thresholds) == 1, 'Threshold list should be either of length 1 or equal length as masks list.'
        
        assign_ops = []
        # new_masks = []
        for index, mask in enumerate(masks):
            abs_w = abs_weights[index]
            threshold = thresholds[min(index, len(thresholds) - 1)]
            new_mask = tf.cast(tf.greater(abs_w, threshold), tf.float32)
            assign_ops.append(tf.assign(mask, new_mask))
            # new_masks.append(new_mask)
        # Assign ops need to be executed for the summaries to capture correct values
        # mask_sparsity_summaries(masks, [m.op.name for m in masks])
    return assign_ops


def conditional_mask_update_op(exclude_scopes,
                               pruning_scheme,
                               global_step,
                               initial_sparsity,
                               final_sparsity,
                               pruning_start_step,
                               pruning_end_step,
                               prune_frequency):
    """
    Conditional mask update ops for gradual pruning.
    https://arxiv.org/abs/1710.01878
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/model_pruning
    
    :param exclude_scopes:
    :param pruning_scheme:
    :param global_step:
    :param initial_sparsity:
    :param final_sparsity:
    :param pruning_start_step:
    :param pruning_end_step:
    :param prune_frequency:
    :return:
    """
    assert pruning_scheme in masked_layer.MAG_PRUNE_MASKS
    if (pruning_end_step - pruning_start_step) % prune_frequency != 0:
        raise ValueError('Pruning end step must be equal to start step added by multiples of frequency.')
    
    def maybe_update_masks():
        with tf.name_scope('mask_update'):
            is_step_within_pruning_range = tf.logical_and(
                tf.greater_equal(global_step, pruning_start_step),
                # If end_pruning_step is negative, keep pruning forever!
                tf.logical_or(
                    tf.less_equal(global_step, pruning_end_step), tf.less(pruning_end_step, 0)))
            is_pruning_step = tf.equal(
                tf.floormod(tf.subtract(global_step, pruning_start_step), prune_frequency), 0)
            is_pruning_step = tf.logical_and(is_step_within_pruning_range, is_pruning_step)
            return is_pruning_step
    
    def mask_update_op():
        current_sparsity = _get_current_sparsity(global_step=global_step,
                                                 initial_sparsity=initial_sparsity,
                                                 final_sparsity=final_sparsity,
                                                 pruning_start_step=pruning_start_step,
                                                 pruning_end_step=pruning_end_step)
        # tf.summary.scalar('sparsity_target', current_sparsity)
        mask_assign_ops = get_mask_assign_ops(
            mask_type=pruning_scheme, sparsity_target=current_sparsity, exclude_scopes=exclude_scopes)
        with tf.control_dependencies(mask_assign_ops):
            # logger.info('Updating masks.')
            return tf.no_op('mask_update')
            # return tf.identity(global_step)
    
    def no_update_op():
        return tf.no_op()
        # return tf.identity(global_step)
    
    return tf.cond(maybe_update_masks(), mask_update_op, no_update_op)


def _get_current_sparsity(global_step,
                          initial_sparsity,
                          final_sparsity,
                          pruning_start_step,
                          pruning_end_step):
    """
    Get current sparsity level for gradual pruning.
    https://arxiv.org/abs/1710.01878
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/model_pruning

    :param global_step:
    :param initial_sparsity:
    :param final_sparsity:
    :param pruning_start_step:
    :param pruning_end_step:
    :return:
    """
    si = initial_sparsity
    sf = final_sparsity
    t = global_step
    t0 = pruning_start_step
    tn = pruning_end_step
    
    p = tf.div(tf.cast(t - t0, tf.float32), tn - t0)
    p = tf.minimum(1.0, tf.maximum(0.0, p))
    st = tf.add(sf, tf.multiply(si - sf, tf.pow(1 - p, 3)))
    return st


def sparsity_loss(sparsity_target,
                  loss_type='L1',
                  exclude_scopes=None):
    """
    Loss for controlling sparsity of Supermasks.
    :param sparsity_target: Desired sparsity rate.
    :param loss_type: The distance metric.
    :param exclude_scopes: Mask scopes to exclude.
    :return: Scalar loss value.
    """
    assert loss_type in LOSS_TYPE, 'Valid loss functions: {}'.format(LOSS_TYPE)
    if loss_type == 'L1':
        loss_fn = _l1_loss
    elif loss_type == 'L2':
        loss_fn = _l2_loss
    elif loss_type == 'hinge_L1':
        loss_fn = _hinge_l1_loss
    else:
        raise ValueError()
    logger.debug('Using mask sparsity loss: `{}`'.format(loss_type))
    sampled_masks, masks = get_masks(exclude_scopes=exclude_scopes)
    if len(masks) == 0:
        return 0.
    total_sparsity = mask_sparsity_summaries(sampled_masks, [m.op.name for m in masks])
    with tf.name_scope('sparsity'):
        # Log average mask value
        mask_vec = tf.concat([tf.reshape(m, [-1]) for m in masks], axis=0)
        mask_av = tf.reduce_mean(mask_vec)
        tf.summary.scalar('mask_average_val', mask_av)
        
        with tf.name_scope('loss'):
            loss = loss_fn(total_sparsity, sparsity_target)
        total_size_np = int(sum([np.prod(_shape(m)) for m in sampled_masks]))
        logger.debug('mask_loss: Total mask size: {:,d}'.format(total_size_np))
    return loss


def _l1_loss(curr, target):
    with tf.name_scope('l1'):
        return tf.abs(tf.subtract(target, curr))
        # return tf.abs(tf.subtract(curr, target))


def _l2_loss(curr, target):
    with tf.name_scope('l2'):
        return tf.squared_difference(curr, target)


def _hinge_l1_loss(curr, target):
    with tf.name_scope('hinge_l1'):
        return tf.nn.relu(tf.subtract(target, curr))


def _get_threshold(abs_weights, sparsity_target, nbins, use_tpu=False):
    with tf.name_scope('get_threshold'):
        max_value = tf.reduce_max(abs_weights)
        cdf_fn = pruning_utils.compute_cdf_from_histogram
        if use_tpu:
            cdf_fn = pruning_utils.compute_cdf
        
        norm_cdf = cdf_fn(abs_weights, [0.0, max_value], nbins=nbins)
        prune_nbins = tf.reduce_sum(tf.cast(tf.less(norm_cdf, sparsity_target), tf.float32))
        threshold = tf.multiply(tf.div(prune_nbins, float(nbins)), max_value)
    return threshold
