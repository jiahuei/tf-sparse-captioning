# -*- coding: utf-8 -*-
"""
Created on 30 Jul 2019 19:26:57

@author: jiahuei
"""

import os
import sys
from tensorflow.contrib.model_pruning.python import pruning_utils

# from tensorflow.contrib.model_pruning.python import pruning
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(CURR_DIR, '..', '..'))
from common import ops_v1 as ops
from common.mask_prune import masked_layer
from common.mask_prune import pruning
from common.mask_prune import sampler
import tensorflow as tf
import numpy as np

_shape = ops.shape
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
SEED = 4896
np.random.seed(SEED)


####


mask_type = masked_layer.MAG_DIST
sps = 0.9
weights_np = [np.random.normal(size=[10]) for i in range(5)]
weights_np[1] += 5.0
print([np.std(w) for w in weights_np])

# Setup input pipeline & Build model
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(SEED)
    weights_tf = [tf.to_float(w) for w in weights_np]
    for w in weights_tf:
        masked_w, _ = masked_layer.generate_masks(kernel=w,
                                                bias=None,
                                                is_training=False,
                                                mask_type=mask_type)
    masks_tf, _ = pruning.get_masks()
    masks_ops = pruning.get_mask_assign_ops(mask_type=mask_type, sparsity_target=sps)
    _w = tf.get_collection('abs_weights')
    
    init_fn = tf.global_variables_initializer()

sess = tf.Session(graph=g)

with sess:
    # Restore model from checkpoint if provided
    sess.run(init_fn)
    sess.run(masks_ops)
    try:
        ww = sess.run(_w)[0]
    except: pass
    w, m = sess.run([weights_tf, masks_tf])
