# -*- coding: utf-8 -*-
"""
Created on 22 Aug 2019 19:40:03

@author: jiahuei
"""
import os
import numpy as np

pjoin = os.path.join
LAYERS = [
    'rnn_initial_state_kernel_mask',
    'basic_lstm_cell_kernel_mask',
    'memory_layer_kernel_mask',
    'value_layer_kernel_mask',
    'query_layer_kernel_mask',
    'MultiHeadAdd_attention_v_mask',
    'rnn_decoder_embedding_map_mask',
    'output_projection_kernel_mask',
    'total_sparsity',
    'total_nnz',
]
LAST_GSTEP = [177000, 531177]


def _valid_csv(dir_name):
    for l in LAYERS:
        if l in dir_name:
            return l
    return None


tb_dump_dir = r'C:\Users\snipe\Documents\GitHub\phd-papers-pruning\resources\tensorboard dumps'
dirs = [pjoin(tb_dump_dir, d) for d in os.listdir(tb_dump_dir)]

# Collect final sparsity values
data = dict(experiments=[])
for d in sorted(dirs):
    if not os.path.isdir(d):
        continue
    exp_name = os.path.basename(d)
    data['experiments'].append(exp_name)
    for f in os.listdir(d):
        layer_name = _valid_csv(f)
        if not layer_name:
            continue
        fpath = pjoin(d, f)
        _, gstep, sparsity = np.genfromtxt(fpath, delimiter=',', skip_header=1)[-1, :]
        assert gstep in LAST_GSTEP
        if layer_name not in data:
            data[layer_name] = []
        data[layer_name].append(sparsity)

# Write output file
fpath = pjoin(tb_dump_dir, 'final_sparsity_values.csv')
out = []
for i, e in enumerate(data['experiments']):
    sps = []
    for l in LAYERS:
        sps.append('{:9.7f}'.format(data[l][i]))
    sps = ','.join(sps)
    out.append(','.join([e, sps]))

headers = ','.join(['Experiments'] + LAYERS)
out = '\n'.join([headers] + out)
with open(fpath, 'w') as f:
    f.write(out)
