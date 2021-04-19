# -*- coding: utf-8 -*-
"""
Created on 15 Jul 2019 19:58:50

@author: jiahuei
"""
from link_dirs import BASE_DIR, pjoin
import argparse
import os
from common.configuration_v1 import load_config


def main(args):
    print('')
    a = args
    default_exp_dir = pjoin(BASE_DIR, 'experiments')
    if a.log_dir == '':
        a.log_dir = default_exp_dir
    if a.inspect_attributes == '':
        print('\nAttribute list is empty.\n')
        return None
    else:
        inspect_attributes = a.inspect_attributes.split(',')
    
    # List experiments
    exp_names = os.listdir(a.log_dir)
    all_run_dirs = []
    for n in exp_names:
        exp_dir = pjoin(a.log_dir, n)
        if os.path.isdir(exp_dir):
            sub_dirs = [pjoin(a.log_dir, n, d) for d in os.listdir(exp_dir)]
            run_dirs = [d for d in sub_dirs if 'infer' not in os.path.split(d)[1]]
            all_run_dirs += run_dirs
    
    # List config files
    # all_cfg_files = []
    # for d in all_run_dirs:
    #     cfg_file = [f for f in os.listdir(d) if 'config' and '.pkl' in f]
    #     assert len(cfg_file) == 1
    #     all_cfg_files.append(pjoin(d, cfg_file[0]))
    all_cfg_files = [pjoin(d, 'config.pkl') for d in all_run_dirs]
    
    # Inspect
    for attr in inspect_attributes:
        print('\nInspecting attribute:  `{}`\n'.format(attr))
        for cpath in all_cfg_files:
            try:
                c = vars(load_config(cpath))
            except IOError:
                continue
            print(os.path.sep.join(cpath.split(os.path.sep)[-3:-1]))
            if attr in c:
                print(c[attr])
            else:
                print('`{}` not found.'.format(attr))
    
    print('\nAttribute inspection completed.\n')


# noinspection PyTypeChecker
def _create_parser():
    _parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    _parser.add_argument(
        '--log_dir', '-l', type=str, default='',
        help='The logging directory.')
    _parser.add_argument(
        '--inspect_attributes', '-a', type=str, default='',
        help='Comma-separated list of attributes to inspect.')
    
    return _parser


if __name__ == '__main__':
    parser = _create_parser()
    main(parser.parse_args())
