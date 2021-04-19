#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:53:58 2017

@author: jiahuei
"""
from link_dirs import CURR_DIR, BASE_DIR, pjoin
import argparse
import os
import logging
import platform
from copy import deepcopy
from time import localtime, strftime
from src import infer_fn_v2 as infer
from src.compat_v2 import update_config
from common import configuration_v1 as cfg
from common.natural_sort import natural_keys as nat_key


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        '--infer_set', type=str, default='test',
        choices=['test', 'valid', 'coco_test', 'coco_valid'],
        help='The split to perform inference on.')
    parser.add_argument(
        '--infer_checkpoints_dir', type=str,
        default=pjoin('mscoco', 'radix_b256_add_LN_softmax_h8_tie_lstm_run_01'),
        help='The directory containing the checkpoint files.')
    parser.add_argument(
        '--infer_checkpoints', type=str, default='all',
        help='The checkpoint numbers to be evaluated. Comma-separated.')
    parser.add_argument(
        '--annotations_file', type=str, default='captions_val2014.json',
        help='The annotations / reference file for calculating scores.')
    parser.add_argument(
        '--dataset_dir', type=str, default='',
        help='Dataset directory.')
    parser.add_argument(
        '--ckpt_prefix', type=str, default='model_compact-',
        help='Prefix of checkpoint names.')
    
    parser.add_argument(
        '--run_inference', type=bool, default=True,
        help='Whether to perform inference.')
    parser.add_argument(
        '--get_metric_score', type=bool, default=True,
        help='Whether to perform metric score calculations.')
    parser.add_argument(
        '--save_attention_maps', type=bool, default=False,
        help='Whether to save attention maps to disk as pickle file.')
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='The gpu number.')
    parser.add_argument(
        '--per_process_gpu_memory_fraction', type=float, default=0.75,
        help='The fraction of GPU memory allocated.')
    parser.add_argument(
        '--verbosity', type=int, default=10, choices=[10, 20])
    
    parser.add_argument(
        '--infer_beam_size', type=int, default=3,
        help='The beam size.')
    parser.add_argument(
        '--infer_length_penalty_weight', type=float, default=0.0,
        help='The length penalty weight used in beam search.')
    parser.add_argument(
        '--infer_max_length', type=int, default=30,
        help='The maximum caption length allowed during inference.')
    parser.add_argument(
        '--batch_size_infer', type=int, default=25,
        help='The batch size.')
    
    args = parser.parse_args()
    return args


def main(args):
    args = deepcopy(args)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    default_exp_dir = pjoin(os.path.dirname(CURR_DIR), 'experiments')
    args.infer_checkpoints_dir = pjoin(default_exp_dir, args.infer_checkpoints_dir)
    args.annotations_file = pjoin(BASE_DIR, 'common', 'coco_caption', 'annotations', args.annotations_file)
    if args.dataset_dir == '':
        args.dataset_dir = pjoin(BASE_DIR, 'datasets', 'mscoco')

    ckpt_prefix = args.ckpt_prefix
    if args.infer_checkpoints == 'all':
        ckpt_files = sorted(os.listdir(args.infer_checkpoints_dir), key=nat_key)
        ckpt_files = [f for f in ckpt_files if ckpt_prefix in f]
        ckpt_files = [f.replace('.index', '') for f in ckpt_files if '.index' in f]
        ckpt_files = [f.replace(ckpt_prefix, '') for f in ckpt_files]
        # if len(ckpt_files) > 20:
        ckpt_files = ckpt_files[-12:]
        args.infer_checkpoints = ckpt_files
    else:
        args.infer_checkpoints = args.infer_checkpoints.split(',')
        if len(args.infer_checkpoints) < 1:
            raise ValueError('`infer_checkpoints` must be either `all` or '
                             'a list of comma-separated checkpoint numbers.')
    
    ###
    
    c = cfg.load_config(pjoin(args.infer_checkpoints_dir, 'config.pkl'))
    c = update_config(c)
    c.__dict__.update(args.__dict__)
    
    save_name = 'b{}_lp{:2.1f}___{}'.format(c.infer_beam_size,
                                            c.infer_length_penalty_weight,
                                            strftime('%m-%d_%H-%M', localtime()))
    set_name = c.infer_set[0] + ''.join(x.title() for x in c.infer_set.split('_'))[1:]  # camelCase
    c.infer_save_path = '_'.join([c.infer_checkpoints_dir, '__infer', set_name, save_name])
    # c.infer_save_path = pjoin(c.infer_checkpoints_dir, '_'.join(['infer', set_name, save_name])
    
    ###
    
    if not os.path.exists(c.infer_save_path):
        os.mkdir(c.infer_save_path)
    
    # Loop through the checkpoint files
    scores_combined = {}
    for ckpt_num in c.infer_checkpoints:
        curr_ckpt_path = pjoin(c.infer_checkpoints_dir, ckpt_prefix + ckpt_num)
        infer.evaluate_model(config=c,
                             curr_ckpt_path=curr_ckpt_path,
                             scores_combined=scores_combined)
        print('\n')


if __name__ == '__main__':
    _args = parse_args()
    logging.basicConfig(level=_args.verbosity)
    logger = logging.getLogger(__name__)
    logger.debug('Python version: {}'.format(platform.python_version()))
    main(_args)
