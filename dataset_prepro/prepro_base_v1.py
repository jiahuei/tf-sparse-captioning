#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:12:25 2017

@author: jiahuei
"""

from ptb_tokenizer import PTBTokenizer
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py
import json
import os
import logging
import time
import re
import io
import common.utils as utils

logger = logging.getLogger(__name__)
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
pjoin = os.path.join


def _convert_split(split, include_restval):
    assert split in ['train', 'val', 'restval', 'valid', 'test']
    if split in 'val':
        return 'valid'
    if include_restval and split == 'restval':
        return 'train'
    assert split in ['train', 'valid', 'test']
    return split


def sort_token_count(token_count_dict):
    """
    Given a dictionary with tokens / words as keys and word count as values,
    return sorted list of tuples (token, count).
    For tokens with same count, they are sorted alphabetically.
    
    :param token_count_dict:
    :return:
    """
    assert isinstance(token_count_dict, dict)
    assert all(isinstance(_, int) for _ in token_count_dict.values())
    token_count = list(token_count_dict.items())
    token_count = sorted(token_count, key=lambda x: x[0])  # break ties using keys (token)
    token_count = sorted(token_count, key=lambda x: x[1], reverse=True)
    return list(token_count)


def load_fasttext_vectors(fname):
    """
    https://fasttext.cc/docs/en/english-vectors.html
    :param fname:
    :return:
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    _data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        _data[tokens[0]] = map(float, tokens[1:])
    return _data


def tokenise(dataset,
             image_id_key='cocoid',
             retokenise=False):
    """
    Tokenise captions (optional), remove non-alphanumerics.
    
    Args:
        dataset: Dictionary object loaded from Karpathy's dataset JSON file.
        image_id_key: String. Used to access `image_id` field.
        retokenise: Boolean. Whether to retokenise the raw captions using 
            Stanford-CoreNLP-3.4.1.
    
    Returns:
        A dictionary with tokenised captions.
        The dict is a list of length-5 dicts:
            filepath : Unicode string.
            image_id : Int
            raw : Length-N list of Unicode strings.
            split : Unicode string.
            tokens : Length-N list of lists. Each list has M Unicode tokens.
    """
    if retokenise:
        logger.info('INFO: Tokenising captions using PTB Tokenizer.\n')
        utils.maybe_download_from_url(
            r'http://central.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar',
            CURR_DIR)
        
        tokenizer = PTBTokenizer()
        
        raw_list = []
        for d in dataset['images']:
            for s in d['sentences']:
                raw_list.append(s['raw'])
        tokenised_cap, tokenised_cap_w_punc = tokenizer.tokenize(raw_list)
        
        tokenised_data = []
        cap_id = 0
        for d in dataset['images']:
            if 'filepath' in d.keys():
                filepath = os.path.join(d['filepath'], d['filename'])
            else:
                filepath = d['filename']
            temp_dict = dict(split=d['split'],
                             filepath=filepath,
                             image_id=d[image_id_key],
                             raw=[],
                             tokens=[])
            for s in d['sentences']:
                temp_dict['raw'].append(s['raw'])
                temp_dict['tokens'].append(
                    [str(w) for w in tokenised_cap[cap_id].split(' ')
                     if w != ''])
                cap_id += 1
            tokenised_data.append(temp_dict)
    else:
        logger.info('Using tokenised captions.\n')
        # pattern = re.compile(r'([^\s\w]|_)+', re.UNICODE)      # matches non-alphanumerics
        pattern = re.compile(r'([^\w]|_)+', re.UNICODE)  # matches non-alphanumerics and whitespaces
        tokenised_data = []
        for d in dataset['images']:
            if 'filepath' in d.keys():
                filepath = os.path.join(d['filepath'], d['filename'])
            else:
                filepath = d['filename']
            temp_dict = dict(split=d['split'],
                             filepath=filepath,
                             image_id=d[image_id_key],
                             raw=[],
                             tokens=[])
            for s in d['sentences']:
                temp_dict['raw'].append(s['raw'])
                temp_list = []
                for w in s['tokens']:
                    w = re.sub(pattern, '', w.lower())
                    if w != '': temp_list.append(w)
                temp_dict['tokens'].append(temp_list)
            tokenised_data.append(temp_dict)
    return tokenised_data


def validate_tokenised_data(tokenised_data, keys=('split', 'tokens')):
    assert isinstance(keys, (tuple, list))
    assert all(isinstance(_, str) for _ in keys)
    assert isinstance(tokenised_data, list)
    assert all(isinstance(_, dict) for _ in tokenised_data)
    for k in keys:
        assert all(k in _ for _ in tokenised_data)


def get_truncate_length(tokenised_dataset,
                        truncate_percentage,
                        include_restval=True):
    """
    Calculates the maximum caption length such that truncated captions makes
    up `truncate_precentage` of the training corpus.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        truncate_percentage: The percentage of truncated captions.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        The maximum caption length.
    """
    validate_tokenised_data(tokenised_dataset)
    
    lengths = {}
    num_captions = 0
    for d in tokenised_dataset:
        split = _convert_split(d['split'], include_restval)
        if split == 'train':
            for s in d['tokens']:
                lengths[len(s)] = lengths.get(len(s), 0) + 1
                num_captions += 1
    truncate_length = 0
    percentage = .0
    for key, value in sorted(lengths.items()):
        if percentage > (100.0 - truncate_percentage):
            truncate_length = key
            break
        percentage += lengths[key] / num_captions * 100
    logger.info('Captions longer than {} words will be truncated.\n'.format(truncate_length))
    return truncate_length


def build_vocab(tokenised_dataset,
                word_count_thres,
                caption_len_thres,
                vocab_size=None,
                include_restval=True,
                pad_value=0,
                include_GO_EOS_tokens=True):
    """
    Builds the word-to-id and id-to-word dictionaries.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        word_count_thres: Threshold for word occurrence. Words that appear
            less than this threshold will be converted to <UNK> token.
        caption_len_thres: Threshold for sentence length in tokens. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
        pad_value: Value assigned to <PAD> token.
    
    Returns:
        Word-to-id and id-to-word dictionaries.
    """
    validate_tokenised_data(tokenised_dataset)
    logger.info('Building vocabulary.\n')
    
    assert pad_value >= -1
    counts = {}
    for d in tokenised_dataset:
        split = _convert_split(d['split'], include_restval)
        if split != 'train':
            continue
        for s in d['tokens']:
            for w in s[:caption_len_thres]:
                counts[w] = counts.get(w, 0) + 1
    
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    
    if vocab_size is None:
        logger.info('Vocab: Filtering out words with count less than {}.\n'.format(
            word_count_thres))
        vocab = [w[1] for w in cw if counts[w[1]] >= word_count_thres]
    else:
        logger.info('Vocab: Generating vocab with fixed size {}.\n'.format(
            vocab_size))
        vocab = [w[1] for w in cw[:vocab_size]]
    # vocab_count = [w for w in cw if counts[w[1]] >= WORD_COUNT_THRES]
    # vocab_inv_freq = [1.0 - (w[0] / float(vocab_count[0][0])) for w in vocab_count]
    # vocab_weights = [0.5 + (f * 1.5) for f in vocab_inv_freq]
    
    vocab = vocab + ['<UNK>']
    if include_GO_EOS_tokens:
        vocab = vocab + ['<GO>', '<EOS>']
    wtoi = {}
    itow = {}
    idx = pad_value
    wtoi['<PAD>'] = idx
    itow[idx] = '<PAD>'
    idx += 1
    for w in vocab:
        wtoi[w] = idx
        itow[idx] = w
        idx += 1
    time.sleep(0.5)
    return wtoi, itow, cw


def tokenised_word_to_txt_V1(tokenised_dataset,
                             caption_len_thres,
                             include_restval=True):
    """
    Builds the train, validation and test lists of texts.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    validate_tokenised_data(tokenised_dataset, ('split', 'filepath', 'tokens'))
    dataset = dict(train=[], valid=[], test=[])
    
    for i, d in enumerate(tqdm(tokenised_dataset,
                               ncols=100, desc='Word-to-txt-V1')):
        split = _convert_split(d['split'], include_restval)
        if split == 'restval': continue
        fp = d['filepath']
        for tokens in d['tokens']:
            sent = ' '.join(tokens[:caption_len_thres])
            sent = '<GO> {} <EOS>'.format(sent)
            sent_out = '{},{}'.format(fp, sent)
            dataset[split].append(sent_out)
    return dataset


def tokenised_word_to_txt_V2(tokenised_dataset,
                             caption_len_thres,
                             include_restval=True):
    """
    Builds the train, validation and test lists of texts.
    
    Args:
        tokenised_dataset: Dictionary from output of `tokenise()`.
        caption_len_thres: Threshold for sentence length in words. Captions
            with longer lengths are truncated.
        include_restval: Boolean. Whether to include `restval` split.
            Only applies to MS-COCO dataset.
    
    Returns:
        `train`, `valid` and `test` dictionaries.
    """
    validate_tokenised_data(tokenised_dataset, ('split', 'filepath', 'tokens'))
    dataset = dict(train=[], valid=[], test=[])
    
    for i, d in enumerate(tqdm(tokenised_dataset,
                               ncols=100, desc='Word-to-txt-V2')):
        split = _convert_split(d['split'], include_restval)
        if split == 'restval': continue
        fp = d['filepath']
        for tokens in d['tokens']:
            tokens = ['<GO>'] + tokens + ['<EOS>']
            sent = ' '.join(tokens[:caption_len_thres + 2])
            sent_out = '{},{}'.format(fp, sent)
            dataset[split].append(sent_out)
    return dataset


def serialise_everything(output_filepath,
                         image_dir,
                         image_size,
                         image_chunk_num,
                         word_to_txt_dict,
                         wtoi,
                         itow):
    assert len(image_size) == 2
    # Assert no overlaps between sets
    train_set = set([s.split(',')[0] for s in word_to_txt_dict['train']])
    valid_set = set([s.split(',')[0] for s in word_to_txt_dict['valid']])
    test_set = set([s.split(',')[0] for s in word_to_txt_dict['test']])
    assert not bool(train_set.intersection(valid_set))
    assert not bool(train_set.intersection(test_set))
    assert not bool(valid_set.intersection(test_set))
    train_set = list(train_set)
    valid_set = list(valid_set)
    test_set = list(test_set)
    
    with h5py.File('{}.h5'.format(output_filepath), 'w') as f:
        sdt = h5py.special_dtype(vlen=str)
        # Store dictionaries
        f.create_dataset('wtoi', data=json.dumps(wtoi))
        f.create_dataset('itow', data=json.dumps(itow))
        
        # Store inference filepaths
        d = f.create_dataset('filenames_valid', (len(valid_set),), dtype=sdt)
        d[:] = valid_set
        d = f.create_dataset('filenames_test', (len(test_set),), dtype=sdt)
        d[:] = test_set
        
        # Create index lookup and add image index
        all_set = train_set + valid_set + test_set
        idx = {}
        for i, p in enumerate(all_set):
            idx[p] = i
        final_dict = {}
        for split in word_to_txt_dict.keys():
            final_dict[split] = []
            for s in word_to_txt_dict[split]:
                fidx = idx[s.split(',')[0]]
                final_dict[split].append('{},{}'.format(fidx, s))
        
        # Store captions used during training
        for split in final_dict.keys():
            captions = final_dict[split]
            d = f.create_dataset(split, (len(captions),), dtype=sdt)
            d[:] = captions
        
        # Store decoded images as NumPy array
        dsize = tuple([len(all_set)] + list(image_size) + [3])
        chunks = tuple([image_chunk_num] + list(image_size) + [3])
        d = f.create_dataset('images', dsize, chunks=chunks, dtype='uint8')
        desc = 'INFO: h5py: Writing images'
        for i, fname in enumerate(tqdm(all_set, ncols=100, desc=desc)):
            fpath = pjoin(image_dir, fname)
            img = Image.open(fpath)
            img = img.resize(image_size, Image.BILINEAR)
            img_arr = np.array(img)
            assert img_arr.dtype == 'uint8'
            err_mssg = 'Corrupted or unsupported image file: `{}`.'
            if len(img_arr.shape) == 3:
                if img_arr.shape[-1] == 3:
                    pass
                elif img_arr.shape[-1] == 1:
                    img_arr = np.concatenate([img_arr] * 3, axis=2)
                else:
                    raise ValueError(err_mssg.format(fpath))
            elif len(img_arr.shape) == 2:
                img_arr = np.stack([img_arr] * 3, axis=2)
            else:
                raise ValueError(err_mssg.format(fpath))
            d[i, :, :, :] = img_arr
    
    logger.info('h5py: Dataset serialisation complete.\n')


def test_h5_file(filepath):
    data = {}
    with h5py.File(filepath, 'r') as f:
        data['wtoi'] = json.loads(f['wtoi'][()])
        data['itow'] = json.loads(f['itow'][()])
        
        data['filenames_valid'] = list(f['filenames_valid'][:])
        data['filenames_test'] = list(f['filenames_test'][:])
        
        data['train'] = list(f['train'][:])
        data['valid'] = list(f['valid'][:])
        data['test'] = list(f['test'][:])
        
        data['images'] = f['images'][:20]
    
    for i in range(10):
        img = Image.fromarray(data['images'][i])
        img.save(pjoin(os.path.split(filepath)[0], 'img_{}.jpg'.format(i)))
    return data


def test_image_files(file_list, log_dir, min_pixels=50 ** 2):
    desc = 'INFO: Testing images ...'
    all_good = True
    for i, fpath in enumerate(tqdm(file_list, ncols=100, desc=desc)):
        img = Image.open(fpath)
        if np.prod(img.size) >= min_pixels:
            pass
        else:
            with open(pjoin(log_dir, 'corrupted_images.txt'), 'a') as f:
                f.write('{}\r\n'.format(fpath))
            all_good = False
    if all_good:
        logger.info('Images all tested.\n')
    return all_good
