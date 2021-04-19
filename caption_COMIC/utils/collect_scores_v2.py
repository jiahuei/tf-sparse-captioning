# -*- coding: utf-8 -*-
"""
Created on 15 Jul 2019 19:58:50

@author: jiahuei
"""
from link_dirs import BASE_DIR, CURR_DIR, pjoin
import argparse
import os
import json
import re
import numpy as np
from time import localtime, strftime
from tqdm import tqdm
from bisect import bisect_left
from common.natural_sort import natural_keys as nat_key
from common import configuration_v1 as cfg

P_NUM = re.compile(r'[0-9][0-9,]+')
pjoin = os.path.join


# noinspection PyTypeChecker
def _create_parser():
    _parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    _parser.add_argument(
        '--log_dir', '-l', type=str, default='',
        help='The logging directory.')
    _parser.add_argument(
        '--collect_runs', '-c', type=str, default='1',
        help='Comma-separated list of runs to collect.')
    _parser.add_argument(
        '--reverse_sort_dirs', '-r', type=bool, default=False,
        help='If True, reverse sort directories.')
    _parser.add_argument(
        '--caption_statistics', '-s', type=bool, default=True,
        help='If True, calculate statistics of captions including percentage of unique captions etc.')
    _parser.add_argument(
        '--verbose', '-v', type=bool, default=False,
        help='If True, print everything.')
    _parser.add_argument(
        '--train_caption_txt', '-t', type=str,
        default='/home/jiahuei/Documents/3_Datasets/MSCOCO_captions/captions/mscoco_train_w5_s20_include_restval.txt',
        help='Training data text file.')
    
    return _parser


def main(args):
    print('')
    a = args
    default_exp_dir = pjoin(BASE_DIR, 'experiments')
    if a.log_dir == '':
        a.log_dir = default_exp_dir
    
    def _should_add(path):
        _is_infer = 'infer' in os.path.split(path)[1]
        if args.collect_runs == 'all':
            return _is_infer
        else:
            runs = ['run_{:02d}'.format(int(r)) for r in args.collect_runs.split(',')]
            return _is_infer and _extract_keys(path)[1] in runs
    
    # List experiments
    exp_dirs = [pjoin(a.log_dir, n) for n in os.listdir(a.log_dir)]
    all_score_dirs = []
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            continue
        sub_dirs = [pjoin(exp_dir, d) for d in os.listdir(exp_dir)]
        score_dirs = [d for d in sub_dirs if _should_add(d)]
        all_score_dirs += score_dirs
    
    # Extract scores
    for sort_checkpoints in [True, False]:
        score_dict = {}
        _loop(all_score_dirs, score_dict, 'valid', sort_checkpoints, args)
        _loop(all_score_dirs, score_dict, 'test', sort_checkpoints, args)
        _write_output_csv(score_dict, args.log_dir, sort_checkpoints, reverse_exps=a.reverse_sort_dirs)
    print('\nScore collection completed.\n')


def _loop(all_score_dirs, score_dict, current_set, sort_checkpoints, args):
    desc = 'Collecting `{}` {} checkpoint sorting'.format(current_set, 'with' if sort_checkpoints else 'without')
    for d in tqdm(sorted(all_score_dirs), desc=desc):
        if current_set not in d:
            continue
        
        exp_name, run, infer_name, datetime = _extract_keys(d)
        score_file = pjoin(d, 'metric_scores.csv')
        if not os.path.isfile(score_file):
            print('WARNING: `{}` does not contain `metric_scores.csv` file.'.format(
                pjoin(exp_name, '___'.join([run, infer_name, datetime]))))
            continue
        
        if args.verbose:
            s = os.path.sep
            print('Processing dir: `{}`'.format(s.join(d.split(s)[-2:])))
        if current_set == 'test':
            valid_name = infer_name.replace('test', 'valid')
            try:
                best_ckpt_num = score_dict[valid_name][exp_name][run]['best_ckpt']
                _, best_score = _get_ckpt(score_file, get_checkpoint_num=best_ckpt_num)
            except KeyError:
                print('WARNING: Validation results not found for: `{}`'.format(
                    pjoin(exp_name, '___'.join([run, infer_name]))))
                continue
        else:
            best_ckpt_num, best_score = _get_ckpt(score_file, sort_checkpoints)
        
        # Get captions stats
        if args.caption_statistics:
            with open(args.train_caption_txt, 'r') as f:
                train_caption = [
                    l.strip().split(',')[1].replace('<GO> ', '').replace(' <EOS>', '') for l in f.readlines()]
            train_caption = set(train_caption)
            # train_caption.sort()
            stats = _get_caption_statistics(train_caption_set=train_caption,
                                            curr_score_dir=d,
                                            checkpoint_num=best_ckpt_num)
        else:
            stats = np.array([-1, -1, -1])
        model_size = _get_model_size(curr_score_dir=d)
        val = dict(name=exp_name, run=run, infer_name=infer_name, datetime=datetime,
                   best_ckpt=best_ckpt_num, best_score=best_score,
                   caption_stats=stats, model_size=model_size)
        
        if infer_name not in score_dict:
            score_dict[infer_name] = {}
        if exp_name not in score_dict[infer_name]:
            score_dict[infer_name][exp_name] = {}
        if run in score_dict[infer_name][exp_name]:
            print('WARNING: `{}` has more than 1 eval results. Keeping latest one.'.format(
                pjoin(exp_name, '___'.join([run, infer_name]))))
        score_dict[infer_name][exp_name][run] = val


def _write_output_csv(sc_dict, log_dir, sort_checkpoints, reverse_exps=False):
    if sort_checkpoints:
        sfx = 'sorted'
    else:
        sfx = 'last'
    
    for infer_name in sorted(sc_dict):
        datetime = strftime('%m-%d_%H-%M-%S', localtime())
        fname = infer_name.replace('infer_', '') + '___{}___{}.csv'.format(sfx, datetime)
        lines = []
        for exp_name in sorted(sc_dict[infer_name], reverse=reverse_exps):
            runs = [sc_dict[infer_name][exp_name][r] for r in sorted(sc_dict[infer_name][exp_name])]
            mean_stats = []
            for i, r in enumerate(runs):
                mean_stats.append(r['caption_stats'])
                if i == 0:
                    name = r['name']
                else:
                    name = '-'
                line = ','.join([name, r['run'], str(r['best_ckpt']),
                                 _score_to_string(r['best_score']),
                                 r['model_size'],
                                 _score_to_string(r['caption_stats']),
                                 r['infer_name'], r['datetime']])
                lines.append(line)
            if len(runs) > 1:
                mean_score = _score_to_string(_get_average([r['best_score'] for r in runs]))
                mean_stats = _score_to_string(_get_average(mean_stats))
                line = ','.join(['-', 'mean', '0', mean_score, mean_stats, 'N/A', 'N/A'])
                lines.append(line)
        with open(pjoin(log_dir, fname), 'w') as f:
            f.write('\r\n'.join(lines))


def _score_to_string(score):
    return ','.join(['{:1.3f}'.format(sc) for sc in list(score)])


def _get_average(list_of_scores):
    scores = np.stack(list_of_scores, axis=0)
    return np.mean(scores, axis=0)


def _get_ckpt(score_file, sort_checkpoints=True, get_checkpoint_num=None):
    scores = np.genfromtxt(score_file, delimiter=',')
    if scores.shape[1] > 3:  # MNIST files have only 3 columns
        scores = scores[:, :-2]
    ckpt_nums, scores = scores[:, 0].astype(np.int64), scores[:, 1:].astype(np.float64)
    
    # Calculate weighted average
    # 2x weightage for B-4, CIDEr, SPICE
    wg = np.array([[1, 1, 1, 2, 1, 1, 2, 2]]).astype(np.float64)
    try:
        scores_wg_av = np.mean(scores * wg, axis=1)
    except ValueError:
        # Give up, take first value lol
        scores_wg_av = scores[:, 0]
    
    if get_checkpoint_num:
        max_idx = int(np.where(ckpt_nums == int(get_checkpoint_num))[0])
        # max_idx = np.where(ckpt_nums == int(get_checkpoint_num))[0]
    else:
        if sort_checkpoints:
            # Get best checkpoint
            max_idx = np.where(scores_wg_av == np.amax(scores_wg_av))
            if len(max_idx[0]) > 1:
                if scores.shape[1] > 6:
                    # For ties, sort by CIDEr
                    max_idx = int(np.argmax(scores[:, 6]))
                else:
                    # MNIST, take last checkpoint
                    max_idx = max_idx[0][-1]
            else:
                max_idx = int(max_idx[0])
        else:
            # Get final checkpoint
            max_idx = ckpt_nums.shape[0] - 1
    
    sparsity_file = pjoin(os.path.split(score_file)[0], 'sparsity_values.csv')
    if os.path.isfile(sparsity_file):
        sparsity = np.genfromtxt(sparsity_file, delimiter=',', skip_header=1)
        
        def _check():
            # noinspection PyTypeChecker
            return sparsity.shape[0] != ckpt_nums.shape[0] or sparsity[max_idx, 0] != ckpt_nums[max_idx]
        
        if _check():
            # Try again without skipping header
            sparsity = np.genfromtxt(sparsity_file, delimiter=',')
        if _check():
            raise ValueError('Checkpoint check failed. {} vs {} for idx {}'.format(
                sparsity[max_idx, 0], ckpt_nums[max_idx], max_idx))
        sparsity = sparsity[max_idx, 1:2]
    else:
        sparsity = [-1]
    score = np.concatenate([sparsity, scores[max_idx], [scores_wg_av[max_idx]]])
    return ckpt_nums[max_idx], score


def _get_caption_statistics(train_caption_set,
                            curr_score_dir,
                            checkpoint_num=None,
                            default_vocab_size=9962):
    # float_str = '{:.3f}'
    # assert isinstance(train_caption_list, list)
    assert isinstance(train_caption_set, set)
    # Try to load vocab size from config
    c = cfg.load_config(pjoin(os.path.dirname(curr_score_dir), 'run_01', 'config.pkl'))
    try:
        vocab_size = len(c.itow)
    except AttributeError:
        vocab_size = default_vocab_size
    
    # Find caption file
    if checkpoint_num is None:
        jsons = [f for f in os.listdir(curr_score_dir) if 'captions___' in f]
        jsons = [j for j in sorted(jsons, key=nat_key)]
        caption_json = pjoin(curr_score_dir, jsons[-1])
    else:
        caption_json = pjoin(curr_score_dir, 'captions___{}.json'.format(checkpoint_num))
    
    # Load captions
    with open(caption_json, 'r') as f:
        captions = json.load(f)
    captions_list = [d['caption'] for d in captions]
    
    # Calculate stats
    appear_in_train = 0
    counts = {}
    caption_length = []
    for caption in captions_list:
        # Unique
        if caption in train_caption_set:
            appear_in_train += 1
        # appear_in_train += binary_search(data_list=train_caption_list, query=caption)
        # Vocab
        caption = caption.split(' ')
        for w in caption:
            counts[w] = counts.get(w, 0) + 1
        # Length
        caption_length.append(len(caption))
    
    vocab_coverage = (len(counts) / (vocab_size - 2)) * 100.  # Exclude <GO> and <EOS>
    average_length = np.mean(caption_length)
    percent_unique = (1. - (appear_in_train / len(captions_list))) * 100.
    return np.array([vocab_coverage, percent_unique, average_length])


def _get_model_size(curr_score_dir):
    # Try to load model size file
    msfp = pjoin(os.path.dirname(curr_score_dir), 'run_01', 'model_size.txt')
    with open(msfp, 'r') as f:
        line = f.readlines()[1]
    model_size = P_NUM.findall(line)
    assert isinstance(model_size, list) and len(model_size) == 1
    return model_size[0].replace(',', '')


def binary_search(data_list, query, lo=0, hi=None):  # can't use data_list to specify default for hi
    # https://stackoverflow.com/questions/212358/binary-search-bisection-in-python
    hi = hi if hi is not None else len(data_list)  # hi defaults to len(data_list)
    pos = bisect_left(a=data_list, x=query, lo=lo, hi=hi)  # find insertion position
    return 1 if pos != hi and data_list[pos] == query else 0  # don't walk off the end


def _extract_keys(score_dirpath):
    exp_dirpath, score_dir = os.path.split(score_dirpath)
    exp_name = os.path.split(exp_dirpath)[1]
    elems = score_dir.split('___')
    if len(elems) == 3:
        run, infer_name, datetime = elems
    elif len(elems) == 2:
        run, infer_name = elems
        datetime = 'none'
    else:
        raise ValueError('Invalid directory name format. Must have at least `run_XX` and `infer_name`.')
    return exp_name, run, infer_name, datetime


if __name__ == '__main__':
    parser = _create_parser()
    main(parser.parse_args())
