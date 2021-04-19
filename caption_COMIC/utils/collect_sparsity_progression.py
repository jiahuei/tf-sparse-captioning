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
]
LAST_GSTEP = [177000, 531177]


def _valid_csv(dir_name):
    for l in LAYERS:
        if l in dir_name:
            return l
    return None


tb_dump_dir = r'C:\Users\snipe\Documents\GitHub\phd-papers-pruning\resources\tensorboard dumps\word_w256_LSTM_r512_xu_REG_1.0e+02_init_5.0_L1_wg_20.0_ann_sps_0.97_constLR'

# Collect sparsity values
data = dict(experiments=os.path.basename(tb_dump_dir))
_gstep = None
for f in os.listdir(tb_dump_dir):
    layer_name = _valid_csv(f)
    if not layer_name:
        continue
    fpath = pjoin(tb_dump_dir, f)
    sparsity = np.genfromtxt(fpath, delimiter=',', skip_header=1)[:, -2:]
    gstep, sparsity = sparsity[:, 0], sparsity[:, 1]
    if _gstep is None:
        _gstep = gstep
    else:
        assert (_gstep == gstep).all()
    data[layer_name] = sparsity
data['gstep'] = _gstep

# Write output file
fpath = pjoin(tb_dump_dir, 'collected_sparsity_values.csv')
out_arr = np.zeros(shape=[len(_gstep), len(LAYERS) + 1])
out_arr[:, 0] = _gstep[:]
for i, l in enumerate(LAYERS):
    out_arr[:, i + 1] = data[l]

out = []
for val in out_arr:
    step = ['{:d}'.format(int(val[0]))]
    sps = ['{:9.7f}'.format(v) for v in val[1:]]
    out.append(','.join(step + sps))
headers = ','.join(['Global step'] + LAYERS)
out = '\n'.join([headers] + out)
with open(fpath, 'w') as f:
    f.write(out)
