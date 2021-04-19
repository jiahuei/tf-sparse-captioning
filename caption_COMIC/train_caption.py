#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:45:51 2017

@author: jiahuei

BUILT AGAINST TENSORFLOW r1.9.0

"""
from link_dirs import BASE_DIR, pjoin
import os
import argparse
import logging
import platform
import tensorflow as tf
import infer_v2 as infer
from src import train_fn_v2 as train
from common.mask_prune import masked_layer
from common import configuration_v1 as cfg
from common import ops_v1 as ops
from common import net_params
from common import utils


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Main
    parser.add_argument(
        '--train_mode', type=str, default='cnn_freeze',
        choices=['cnn_freeze', 'cnn_finetune', 'scst', 'mnist', 'autoencoder'],
        help='Str. The training regime.')
    
    # Data & Checkpoint
    parser.add_argument(
        '--log_root', type=str, default='',
        help='The logging root directory.')
    parser.add_argument(
        '--dataset_dir', type=str, default='',
        help='The dataset directory.')
    parser.add_argument(
        '--dataset_file_pattern', type=str, default='mscoco_{}_w5_s20_include_restval',
        help='The dataset text files naming pattern.')
    parser.add_argument(
        '--glove_filepath', type=str, default='',
        help='The file path of GloVe embedding.')
    parser.add_argument(
        '--checkpoint_path', type=str, default='',
        help='The checkpoint path. Can be a dir containing CNN ckpts, dir containing model ckpts, ckpt file path.')
    parser.add_argument(
        '--checkpoint_exclude_scopes', type=str, default='',
        help='The scopes to exclude when restoring from checkpoint.')
    
    # Model
    parser.add_argument(
        '--use_glove_embeddings', type=bool, default=False,
        help='If True, initialise word embedding matrix using GloVe embeddings.')
    
    # SCST
    parser.add_argument(
        '--scst_beam_size', type=int, default=7,
        help='Beam size for SCST sampling.')
    parser.add_argument(
        '--scst_weight_ciderD', type=float, default=1.0,
        help='Weight for CIDEr-D metric during SCST training.')
    parser.add_argument(
        '--scst_weight_bleu', type=ops.convert_float_csv, default='0,0,0,0',
        help='Weight for BLEU metrics during SCST training.')
    
    # Misc
    parser.add_argument(
        '--name', type=str, default='lstm',
        help='The logging name.')
    parser.add_argument(
        '--run', type=int, default=1,
        help='The run number.')
    
    train.add_args(parser)
    args = parser.parse_args()
    return args


def main(args, run_inference=None):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    rand_seed = [0,
                 48964896,
                 88888888,
                 123456789]
    try:
        rand_seed = rand_seed[args.run]
    except IndexError:
        rand_seed = args.run
    
    if args.attn_size is None or not isinstance(args.attn_size, (float, int)) or args.attn_size <= 1:
        args.attn_size = args.rnn_size
    
    if args.legacy:
        logger.info('LEGACY mode enabled. Some arguments will be overridden.')
        args.cnn_name = 'inception_v1'
        args.cnn_input_size = '224,224'
        args.cnn_input_augment = True
        args.cnn_fm_attention = 'Mixed_4f'
        args.rnn_name = 'LSTM'
        args.rnn_size = 512
        args.rnn_word_size = 256
        args.rnn_init_method = 'project_hidden'
        args.rnn_keep_prob = 0.65
        args.rnn_recurr_dropout = False
        args.attn_context_layer = False
        args.attn_alignment_method = 'add_LN'
        args.attn_probability_fn = 'softmax'
        args.attn_keep_prob = 1.0
        args.lr_start = 1e-3
        args.lr_end = 2e-4
        args.lr_reduce_every_n_epochs = 4
        args.cnn_grad_multiplier = 1.0
        args.initialiser = 'xavier'
        args.optimiser = 'adam'
        args.batch_size_train = 32
        args.adam_epsilon = 1e-6
    
    ###
    
    ## Log name
    
    dataset = args.dataset_file_pattern.split('_')[0]
    if args.log_root == '':
        args.log_root = pjoin(BASE_DIR, 'experiments')
    log_root = pjoin(args.log_root, dataset + '_v3')
    
    if args.dataset_dir == '':
        args.dataset_dir = pjoin(BASE_DIR, 'datasets', dataset)
    
    if not os.path.isfile(args.glove_filepath):
        args.glove_filepath = pjoin(BASE_DIR, 'pretrained', 'glove.6B.300d.txt')

    if args.supermask_type:
        if args.supermask_sparsity_weight < 0:
            if 'masked' in args.cnn_name:
                args.supermask_sparsity_weight = max(5., 1.5 / (1 - args.supermask_sparsity_target))
            else:
                args.supermask_sparsity_weight = max(5., 0.5 / (1 - args.supermask_sparsity_target))
        if args.supermask_type and args.supermask_sparsity_target > 0:
            args.rnn_keep_prob = 0.89
            args.attn_keep_prob = 0.97
        if args.supermask_type in masked_layer.MAG_HARD:
            args.max_epoch = 10

    name = train.get_log_name(args)
    if args.name:
        name = '{}_{}'.format(name, args.name)
    
    dec_dir = pjoin(log_root, '{}'.format(name), 'run_{:02d}'.format(args.run))
    cnnft_dir = pjoin(log_root, '{}_cnnFT'.format(name), 'run_{:02d}'.format(args.run))
    log_path = dec_dir
    train_fn = train.train_caption_xe
    
    if args.train_mode == 'cnn_freeze':
        assert args.freeze_scopes == ['Model/encoder/cnn']
        _ckpt = args.checkpoint_path
        if os.path.isfile(_ckpt + '.index') or os.path.isfile(_ckpt) or tf.train.latest_checkpoint(_ckpt):
            pass
        else:
            # Maybe download weights
            net = net_params.get_net_params(args.cnn_name, ckpt_dir_or_file=args.checkpoint_path)
            utils.maybe_get_ckpt_file(net)
            args.checkpoint_path = net['ckpt_path']
    
    elif args.train_mode == 'cnn_finetune':
        # CNN fine-tune
        if args.legacy:
            raise NotImplementedError
        # if not os.path.exists(dec_dir):
        #     raise ValueError('Decoder training log path not found: {}'.format(dec_dir))
        args.lr_start = 1e-3
        args.max_epoch = 10
        args.freeze_scopes = None
        # args.checkpoint_path = dec_dir
        log_path = cnnft_dir
    
    elif args.train_mode == 'scst':
        # SCST fine-tune (after CNN fine-tune)
        if args.legacy:
            raise NotImplementedError
        # if not os.path.exists(cnnft_dir):
        #     raise ValueError('CNN finetune log path not found: {}'.format(cnnft_dir))
        args.batch_size_train = 10
        args.lr_start = 1e-3
        args.max_epoch = 10 if dataset == 'mscoco' else 3
        args.freeze_scopes = ['Model/encoder/cnn']
        # args.checkpoint_path = cnnft_dir
        scst = 'b{}C{}B{}'.format(
            args.scst_beam_size,
            args.scst_weight_ciderD,
            args.scst_weight_bleu[-1])
        scst_dir = pjoin(log_root, '{}_cnnFT_SCST_{}'.format(name, scst), 'run_{:02d}'.format(args.run))
        log_path = scst_dir
        train_fn = train.train_caption_scst
    
    elif args.train_mode == 'mnist':
        train_fn = train.train_rnn_mnist
        args.batch_size_train = 100
        args.batch_size_eval = 100
        args.lr_start = 0.1
        args.max_epoch = 60
        args.optimiser = 'sgd'
        args.checkpoint_path = None
        log_path = dec_dir
    
    # elif args.train_mode == 'autoencoder':
    #     train_fn = train.train_caption_ae
    #     args.checkpoint_path = None
    #     log_path = dec_dir
    
    ###
    
    defaults = dict(
        rnn_layers=1,
        rnn_keep_in=args.rnn_keep_prob,
        rnn_keep_out=args.rnn_keep_prob,
        
        max_saves=12 if args.train_mode != 'mnist' else 3,
        num_logs_per_epoch=100 if args.train_mode != 'mnist' else 5,
        per_process_gpu_memory_fraction=None,
        
        rand_seed=rand_seed,
        add_image_summaries=False,
        add_vars_summaries=False,
        add_grad_summaries=False,
        
        log_path=log_path,
        save_path=pjoin(log_path, 'model'),
        resume_training=args.resume_training and os.path.exists(log_path),
    )
    
    del args.rnn_keep_prob
    defaults.update(vars(args))
    config = cfg.Config(**defaults)
    config.overwrite_safety_check(overwrite=args.resume_training)
    
    ###
    
    train.try_to_train(train_fn=train_fn, config=config, try_block=False)
    
    ###
    
    if run_inference in ['test', 'valid', 'coco_test', 'coco_valid'] and args.train_mode not in ['mnist',
                                                                                                 'autoencoder']:
        args.infer_set = run_inference
        args.infer_checkpoints_dir = log_path
        args.infer_checkpoints = 'all'
        args.ckpt_prefix = 'model_compact-'
        if 'coco' in dataset:
            args.annotations_file = 'captions_val2014.json'
        elif 'insta' in dataset:
            args.annotations_file = 'insta_testval_clean.json'
        else:
            raise NotImplementedError('Invalid dataset: {}'.format(dataset))
        args.run_inference = True
        args.get_metric_score = True
        args.save_attention_maps = False
        args.per_process_gpu_memory_fraction = 0.75
        args.infer_beam_size = 3
        args.infer_length_penalty_weight = 0.
        args.infer_max_length = 30
        args.batch_size_infer = 25
        infer.main(args)
        args.infer_set = 'test'
        infer.main(args)


if __name__ == '__main__':
    _args = parse_args()
    logging.basicConfig(level=_args.verbosity)
    logger = logging.getLogger(__name__)
    logger.debug('Python version: {}'.format(platform.python_version()))
    main(_args, run_inference='valid')
