#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:55:32 2017

@author: jiahuei

V9
"""
from link_dirs import CURR_DIR, BASE_DIR, up_dir, pjoin
from tqdm import tqdm
import os
import logging
import copy
import argparse
import json
import re
import random
import prepro_base_v1 as prepro
import common.utils as utils


# wtoi_file = 'insta_wtoi_w5_s15_split.json'
# itow_file = 'insta_itow_w5_s15_split.json'

# For tokenization
# https://github.com/cesc-park/attend2u/blob/c1185e550c72f71daa74a6ac95791cbf33363b27/scripts/generate_dataset.py
try:
    # UCS-4
    EMOTICON = re.compile('(([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF]))')
except Exception as e:
    # UCS-2
    EMOTICON = re.compile(
        '(([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF]))')
NOT_EMOTICON = re.compile(r'(\\U([0-9A-Fa-f]){8})|(\\u([0-9A-Fa-f]){4})')


def tokenize(sentence):
    """Tokenize a sentence"""
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    
    sentence = sentence.replace('#', ' #')
    sentence = sentence.replace('@', ' @')
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.lower()
    sentence = re.sub(r'@[a-zA-Z0-9._]+', '@username', sentence)  # change username
    sentence = EMOTICON.sub(r'@@byeongchang\1 ', sentence)
    sentence = sentence.encode('unicode-escape').decode('ascii')  # for emoticons
    sentence = re.sub(r'@@byeongchang\\', '@@byeongchang', sentence)
    sentence = NOT_EMOTICON.sub(r' ', sentence)
    sentence = re.sub(r"[\-_]", r"-", sentence)  # incoporate - and _
    sentence = re.sub(r"([!?,\.\"])", r" ", sentence)  # remove duplicates on . , ! ?
    sentence = re.sub(r"(?<![a-zA-Z0-9])\-(?![a-zA-Z0-9])", r"",
                      sentence)  # remove - if there is no preceed or following
    sentence = ' '.join(re.split(r'[^a-zA-Z0-9#@\'\-]+', sentence))
    sentence = re.sub(r'@@byeongchang', r' \\', sentence)
    return sentence.split()


def tokenize_all(train_json, test1_json):
    """
    Tokenize sentences in raw dataset

    Args:
        train_json, test1_json: raw json object
        key: 'caption' or 'tags'
    """
    
    # print("\nINFO: Tokenising captions.\n")
    tokenised_data = []
    # Train data
    for user_id, posts in tqdm(sorted(train_json.items()),
                               ncols=100, desc='Tokenising train data'):
        for post_id, post in sorted(posts.items()):
            img_id = '{}_@_{}'.format(user_id, post_id)
            temp_dict = dict(split='train',
                             filepath=pjoin('images', img_id),
                             image_id=img_id,
                             raw=[post['caption']],
                             tokens=[tokenize(post['caption'])])
            tokenised_data.append(temp_dict)
    
    # Validation data
    random.seed(4896)
    random.shuffle(tokenised_data)
    for i in range(2000):
        tokenised_data[i]['split'] = 'val'
    
    # Test1 data
    for user_id, posts in tqdm(sorted(test1_json.items()),
                               ncols=100, desc='Tokenising test1 data'):
        for post_id, post in sorted(posts.items()):
            img_id = '{}_@_{}'.format(user_id, post_id)
            temp_dict = dict(split='test',
                             filepath=pjoin('images', img_id),
                             image_id=img_id,
                             raw=[post['caption']],
                             tokens=[tokenize(post['caption'])])
            tokenised_data.append(temp_dict)
    return tokenised_data


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dataset_dir', type=str, default='')
    parser.add_argument(
        '--output_prefix', type=str, default='insta')
    parser.add_argument(
        '--word_count_thres', type=int, default=5)
    parser.add_argument(
        '--caption_len_thres', type=int, default=15)
    parser.add_argument(
        '--vocab_size', type=int, default=25595)
    parser.add_argument(
        '--pad_value', type=int, default=-1)
    parser.add_argument(
        '--wtoi_file', type=str, default=None)
    parser.add_argument(
        '--itow_file', type=str, default=None)
    parser.add_argument(
        '--verify_images', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    args = parse_args()
    if args.vocab_size < 1 or isinstance(args.vocab_size, str):
        args.vocab_size = None
    
    if args.dataset_dir == '':
        dset_dir = pjoin(os.path.dirname(CURR_DIR), 'datasets', 'insta')
    else:
        dset_dir = args.dataset_dir
    out_path = pjoin(dset_dir, 'captions')
    cap_train_json_path = pjoin(dset_dir, 'json', 'insta-caption-train.json')
    cap_test1_json_path = pjoin(dset_dir, 'json', 'insta-caption-test1.json')
    
    ### Get the caption JSON files ###
    json_exists = (os.path.isfile(cap_train_json_path) and
                   os.path.isfile(cap_test1_json_path))
    tgz_path = pjoin(dset_dir, 'json.tar.gz')
    if json_exists:
        logger.info('Found exising json files.')
    else:
        utils.maybe_download_from_google_drive(
            r'0B3xszfcsfVUBdG0tU3BOQWV0a0E',
            tgz_path,
            file_size=669 * 1024 ** 2)
        utils.extract_tar_gz(tgz_path)
        # os.remove(tgz_path)
    
    ### Read the raw JSON file ###
    
    print('\nINFO: Reading JSON files.\n')
    with open(cap_train_json_path, 'r') as f:
        cap_train_json = json.load(f)
    with open(cap_test1_json_path, 'r') as f:
        cap_test1_json = json.load(f)
    
    ### Tokenize all ###
    tokenised_insta = tokenize_all(cap_train_json, cap_test1_json)
    tokenised_insta_copy = copy.deepcopy(tokenised_insta)
    print('')
    
    ### Build vocabulary ###
    
    cw = None
    build_vocab = args.wtoi_file is None or args.itow_file is None
    if build_vocab:
        wtoi, itow, cw = prepro.build_vocab(tokenised_insta,
                                            args.word_count_thres,
                                            args.caption_len_thres,
                                            vocab_size=args.vocab_size,
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
    
    tokenised_insta = prepro.tokenised_word_to_txt_V1(tokenised_insta,
                                                      args.caption_len_thres,
                                                      include_restval=False)
    
    print('\nINFO: Example captions:')
    for j in range(5):
        print(tokenised_insta['train'][j])
    print('\n')
    
    ### Output files ###
    
    if args.output_prefix is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        if args.vocab_size is None:
            suffix = 'w{}_s{}'.format(args.word_count_thres, args.caption_len_thres)
        else:
            suffix = 'v{}_s{}'.format(args.vocab_size, args.caption_len_thres)
        
        if cw is not None:
            wc = ['{},{}'.format(w, c) for c, w in cw]
            with open(pjoin(out_path, '{}_word_freq.csv').format(args.output_prefix), 'w') as f:
                f.write('\r\n'.join(wc))
        
        for split in tokenised_insta.keys():
            filename = '{}_{}_{}.txt'.format(args.output_prefix, split, suffix)
            with open(pjoin(out_path, filename), 'w') as f:
                f.write('\r\n'.join(tokenised_insta[split]))
        
        # Assert no overlaps between sets
        train_set = set([s.split(',')[0] for s in tokenised_insta['train']])
        valid_set = set([s.split(',')[0] for s in tokenised_insta['valid']])
        test_set = set([s.split(',')[0] for s in tokenised_insta['test']])
        assert not bool(train_set.intersection(valid_set))
        assert not bool(train_set.intersection(test_set))
        assert not bool(valid_set.intersection(test_set))
        
        # Write train file list
        # with open(pjoin(OUT_PATH, 'filenames_train.txt'), 'w') as f:
        #    f.write('\r\n'.join(list(train_set)))
        
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
        
        coco_annot_dir = pjoin(BASE_DIR, 'common', 'coco_caption', 'annotations')
        # Generate COCO style annotation file (raw)
        test_ann = dict(images=[],
                        info='',
                        type='captions',
                        annotations=[],
                        licenses='')
        
        for d in tokenised_insta_copy:
            if d['split'] not in ['test', 'val']:
                continue
            test_ann['images'].append({'id': d['image_id']})
            test_ann['annotations'].append(
                {'caption': d['raw'][0],
                 'id': 0,
                 'image_id': d['image_id']})
        
        with open(pjoin(coco_annot_dir, 'insta_testval_raw.json'), 'w') as f:
            json.dump(test_ann, f)
        
        # Generate COCO style annotation file (without emojis)
        test_ann = dict(images=[],
                        info='',
                        type='captions',
                        annotations=[],
                        licenses='')
        
        for d in tokenised_insta_copy:
            if d['split'] not in ['test', 'val']:
                continue
            test_ann['images'].append({'id': d['image_id']})
            test_ann['annotations'].append(
                {'caption': ' '.join(d['tokens'][0]),
                 'id': 0,
                 'image_id': d['image_id']})
        
        with open(pjoin(coco_annot_dir, 'insta_testval_clean.json'), 'w') as f:
            json.dump(test_ann, f)
        
        logger.info('Saved output text files.\n')
    
    
    ### Get the image files ###
    def _check_img_exists():
        logger.info('Listing existing image files.')
        img_all = train_set.union(valid_set).union(test_set)
        ex = []
        if os.path.exists(pjoin(dset_dir, 'images')):
            ex = os.listdir(pjoin(dset_dir, 'images'))
            ex = [pjoin('images', i) for i in ex]
        ex = set(ex)
        return len(ex.intersection(img_all)) == len(img_all), list(ex)
    
    
    img_exists, img_list = _check_img_exists()
    if img_exists:
        logger.info('Found existing image files.')
    else:
        tgz_path = pjoin(dset_dir, 'images.tar.gz')
        utils.maybe_download_from_google_drive(
            r'0B3xszfcsfVUBVkZGU2oxYVl6aDA',
            tgz_path,
            file_size=20 * 1024 ** 3)
        utils.extract_tar_gz(tgz_path)
        # os.remove(tgz_path)
        img_exists, img_list = _check_img_exists()
        if not img_exists:
            raise ValueError('INFO: Image download incomplete. Please download again by re-running the script.')
    
    if args.verify_images:
        prepro.test_image_files(file_list=[pjoin(dset_dir, i) for i in img_list], log_dir=dset_dir)
