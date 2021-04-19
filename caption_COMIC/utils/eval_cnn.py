# -*- coding: utf-8 -*-
"""
Created on 03 Mar 2020 14:52:26

@author: jiahuei
"""
from link_dirs import BASE_DIR, pjoin
import os
import platform
import argparse
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import localtime, strftime
from common import utils
from common.net_params import get_net_params
from common.imagenet.eval_imagenet import eval_validation


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        '--imagenet_valid_dir', type=str, default='/master/datasets/imagenet/val',
        help='ImageNet validation set directory.')
    parser.add_argument(
        '--cnn_list', type=str,
        default='inception_v1,mobilenet_v1,mobilenet_v1_075,mobilenet_v1_050,mobilenet_v1_025',
        help='ImageNet validation set directory.')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='The output directory.')
    parser.add_argument(
        '--batch_size', type=int, default=50,
        help='The batch size.')
    parser.add_argument(
        '--gpu', type=str, default='1',
        help='The gpu number.')
    parser.add_argument(
        '--visualise', type=bool, default=False,
        help='Whether to sample some results to visualise.')

    return parser.parse_args()


def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.output_dir is None:
        output_dir = CURR_DIR
    else:
        output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    time_str = strftime('%Y%m%d_%H%M%S', localtime())
    
    for cnn_name in args.cnn_list.split(','):
        net_params = get_net_params(cnn_name, ckpt_dir_or_file=output_dir)
        utils.maybe_get_ckpt_file(net_params)
        top1, top5 = eval_validation(
            valid_dir=args.imagenet_valid_dir,
            cnn_name=cnn_name,
            checkpoint_path=net_params['ckpt_path'],
            batch_size=args.batch_size,
            visualise=args.visualise,
        )
        top1 *= 100
        top5 *= 100
        print('Accuracies:    Top-1: {:4.1f}    Top-5: {:4.1f}'.format(top1, top5))
        outputs = '{},{:4.1f},{:4.1f}\n'.format(cnn_name, top1, top5)
        
        with open(pjoin(output_dir, 'accuracies_{}.txt'.format(time_str)), 'a') as f:
            f.write(outputs)


if __name__ == '__main__':
    assert platform.python_version().startswith('3'), 'Only Python 3 is supported.'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main(parse_args())
