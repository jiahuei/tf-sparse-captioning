# -*- coding: utf-8 -*-
"""
Created on 10 Jul 2019 13:55:58

@author: jiahuei
"""
from copy import deepcopy
from common import ops_v1 as ops


def update_config(old_config):
    config = deepcopy(old_config)
    
    # update params
    _update_params(config, 'attn_size', old_config.rnn_size)
    # _update_params(config, 'radix_max_word_len', len(ops.number_to_base(len(config.wtoi), config.radix_base)))
    if config.supermask_type == 'mag_gradual':
        config.supermask_type = 'mag_grad_uniform'
    return config


def _update_params(config, key, value, replace_existing_value=False):
    try:
        config.__dict__[key]
        if replace_existing_value:
            config.__dict__[key] = value
    except KeyError:
        config.__dict__[key] = value


