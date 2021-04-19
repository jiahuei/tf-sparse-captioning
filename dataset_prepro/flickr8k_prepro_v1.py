#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:34:56 2017

@author: jiahuei

V9
"""
from link_dirs import CURR_DIR, BASE_DIR, up_dir, pjoin
import os
import json
import argparse
import logging
import prepro_base_v1 as prepro
import common.utils as utils

JSON_FILE = 'dataset_flickr8k.json'


# wtoi_file = 'coco_wtoi_w5_s20_include_restval.json'
# itow_file = 'coco_itow_w5_s20_include_restval.json'


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dataset_dir', type=str, default='')
    parser.add_argument(
        '--output_prefix', type=str, default='flickr8k')
    parser.add_argument(
        '--retokenise', type=bool, default=False)
    parser.add_argument(
        '--word_count_thres', type=int, default=5)
    parser.add_argument(
        '--caption_len_thres', type=int, default=20)
    parser.add_argument(
        '--pad_value', type=int, default=-1)
    parser.add_argument(
        '--wtoi_file', type=str, default=None)
    parser.add_argument(
        '--itow_file', type=str, default=None)
    parser.add_argument(
        '--verify_images', type=bool, default=False)
    
    return parser


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = create_parser()
    args = parser.parse_args()
    
    if args.dataset_dir == '':
        dset_dir = pjoin(os.path.dirname(CURR_DIR), 'datasets', 'flickr30k')
    else:
        dset_dir = args.dataset_dir
    out_path = pjoin(dset_dir, 'captions')
    json_path = pjoin(dset_dir, JSON_FILE)
    
    ### Get the caption JSON files ###
    if os.path.isfile(json_path):
        logger.info('Found file: `{}`'.format(JSON_FILE))
    else:
        zip_path = utils.maybe_download_from_url(
            r'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip',
            dset_dir)
        utils.extract_zip(zip_path)
        # os.remove(zip_path)
    
    ### Read the raw JSON file ###
    
    with open(json_path, 'r') as f:
        dataset_f8k = json.load(f)
    
    ### Tokenise captions ###
    
    tokenised_f8k = prepro.tokenise(dataset_f8k,
                                    image_id_key='imgid',
                                    retokenise=args.retokenise)
    
    ### Build vocabulary ###
    
    cw = None
    build_vocab = args.wtoi_file is None or args.itow_file is None
    if build_vocab:
        wtoi, itow, cw = prepro.build_vocab(tokenised_f8k,
                                            args.word_count_thres,
                                            args.caption_len_thres,
                                            vocab_size=None,
                                            include_restval=False,
                                            pad_value=args.pad_value,
                                            include_GO_EOS_tokens=True)
    else:
        logger.info('Reusing provided vocabulary.\n')
        with open(os.path.join(out_path, args.wtoi_file), 'r') as f:
            wtoi = json.load(f)
        with open(os.path.join(out_path, args.itow_file), 'r') as f:
            itow = json.load(f)
    
    ### Convert tokenised words to text files ###
    
    tokenised_f8k = prepro.tokenised_word_to_txt_V1(tokenised_f8k,
                                                    args.caption_len_thres,
                                                    include_restval=False)
    
    print('\nINFO: Example captions:')
    for j in range(5):
        print(tokenised_f8k['train'][j])
    print('\n')
    
    ### Output files ###
    
    if args.output_prefix is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        suffix = ['w{:d}_s{:d}'.format(args.word_count_thres, args.caption_len_thres)]
        if args.retokenise:
            suffix.append('retokenised')
        suffix = '_'.join(suffix)
        
        if cw is not None:
            wc = ['{},{}'.format(w, c) for c, w in cw]
            with open(pjoin(out_path, '{}_word_freq.csv').format(args.output_prefix), 'w') as f:
                f.write('\r\n'.join(wc))
        
        for split in tokenised_f8k.keys():
            filename = '{}_{}_{}.txt'.format(args.output_prefix, split, suffix)
            with open(pjoin(out_path, filename), 'w') as f:
                f.write('\r\n'.join(tokenised_f8k[split]))
        
        # Assert no overlaps between sets
        train_set = set([s.split(',')[0] for s in tokenised_f8k['train']])
        valid_set = set([s.split(',')[0] for s in tokenised_f8k['valid']])
        test_set = set([s.split(',')[0] for s in tokenised_f8k['test']])
        assert not bool(train_set.intersection(valid_set))
        assert not bool(train_set.intersection(test_set))
        assert not bool(valid_set.intersection(test_set))
        
        # Write validation file list
        with open(pjoin(out_path, 'filenames_valid.txt'), 'w') as f:
            f.write('\r\n'.join(list(valid_set)))
        
        # Write test file list
        with open(pjoin(out_path, 'filenames_test.txt'), 'w') as f:
            f.write('\r\n'.join(list(test_set)))
        
        if build_vocab:
            with open(pjoin('{}', '{}_wtoi_{}.json').format(
                    out_path, args.output_prefix, suffix), 'w') as f:
                json.dump(wtoi, f)
            with open(pjoin('{}', '{}_itow_{}.json').format(
                    out_path, args.output_prefix, suffix), 'w') as f:
                json.dump(itow, f)
        
        logger.info('Saved output text files.\n')
    
    # ### Get the image files ###
    # def _check_img_exists():
    #     logger.info('Listing existing image files.')
    #     img_all = train_set.union(valid_set).union(test_set)
    #     trpath = pjoin(dset_dir, 'train2014')
    #     vpath = pjoin(dset_dir, 'val2014')
    #     ttpath = pjoin(dset_dir, 'test2014')
    #     extr = exv = extt = []
    #     if os.path.exists(trpath):
    #         extr = os.listdir(trpath)
    #         extr = [pjoin('train2014', i) for i in extr]
    #     if os.path.exists(vpath):
    #         exv = os.listdir(vpath)
    #         exv = [pjoin('val2014', i) for i in exv]
    #     if os.path.exists(ttpath):
    #         extt = os.listdir(ttpath)
    #         extt = [pjoin('test2014', i) for i in extt]
    #     ex = set(extr + exv)
    #     exists = len(ex.intersection(img_all)) == len(img_all)
    #     exists = exists and len(extt) == 40775
    #     return exists, list(ex) + extt
    #
    #
    # img_exists, img_list = _check_img_exists()
    # if img_exists:
    #     logger.info('Found existing image files.')
    # else:
    #     zip_path = utils.maybe_download_from_url(
    #         r'http://images.cocodataset.org/zips/train2014.zip',
    #         dset_dir)
    #     utils.extract_zip(zip_path)
    #     # os.remove(zip_path)
    #     zip_path = utils.maybe_download_from_url(
    #         r'http://images.cocodataset.org/zips/val2014.zip',
    #         dset_dir)
    #     utils.extract_zip(zip_path)
    #     # os.remove(zip_path)
    #     zip_path = utils.maybe_download_from_url(
    #         r'http://images.cocodataset.org/zips/test2014.zip',
    #         dset_dir)
    #     utils.extract_zip(zip_path)
    #     # os.remove(zip_path)
    #     img_exists, img_list = _check_img_exists()
    #     if not img_exists:
    #         raise ValueError('INFO: Image download incomplete. Please download again by re-running the script.')
    #
    # if args.verify_images:
    #     prepro.test_image_files(file_list=[pjoin(dset_dir, i) for i in img_list], log_dir=dset_dir)
