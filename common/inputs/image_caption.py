# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:59:29 2017

@author: jiahuei

Currently Manager adds the following attributes to the Config object:
'vocab_size', 'max_step', 'split_sizes', 'itow', 'wtoi'

"""
import tensorflow as tf
import numpy as np
import os
import json
import string
import logging
import random
from itertools import chain
from common.nets import nets_factory
from common.inputs.preprocessing import preprocessing_factory as prepro_factory
from common import ops_v1 as ops

logger = logging.getLogger(__name__)
_shape = ops.shape
pjoin = os.path.join
# slim = tf.contrib.slim

TOKEN_RADIX = 'radix'
TOKEN_WORD = 'word'
TOKEN_CHAR = 'char'
VALID_TOKEN_TYPE = [TOKEN_RADIX, TOKEN_WORD, TOKEN_CHAR]

MODE_TRAIN = 'train'
MODE_EVAL = 'eval'
MODE_INFER = 'infer'
VALID_MODE = [MODE_TRAIN, MODE_EVAL, MODE_INFER]

SET_TEST = 'test'
SET_VALID = 'valid'
SET_COCO_TEST = 'coco_test'
SET_COCO_VALID = 'coco_valid'
VALID_INFER_SET = [SET_TEST, SET_VALID, SET_COCO_TEST, SET_COCO_VALID]

ERROR_UNKNOWN_DATASET = 'Unknown dataset.'


class CaptionInput:
    """ Input Manager object."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """
        logger.debug('Using `common.inputs.image_caption` (v3).')
        self.config = c = config
        
        # Determine the input size of image CNN
        im_net = nets_factory.networks_map[c.cnn_name]
        s = c.cnn_input_size
        if isinstance(s, list) and len(s) == 2 and 0 not in s:
            logger.info('Using specified CNN input size: {}.'.format(s))
        else:
            if hasattr(im_net, 'default_image_size'):
                c.cnn_input_size = [im_net.default_image_size] * 2
                logger.info('Using default CNN input size: {}.'.format(c.cnn_input_size))
            else:
                raise ValueError('Unable to retrieve default image size.')
        
        # Add new info to config
        c.split_sizes = {}
        self.is_inference = is_inference
        random.seed(c.rand_seed)
        
        # Read vocab files
        assert c.token_type in VALID_TOKEN_TYPE
        self._set_vocab()
        
        # Setup input pipelines
        # with tf.device("/cpu:0"):
        if is_inference:
            self.batch_infer = self._batch_setup('infer')
        else:
            self.batch_train = self._batch_setup('train')
            self.batch_eval = self._batch_setup('valid')
        logger.info('Input pipelines setup complete.')
    
    def _set_vocab(self):
        c = self.config
        
        if self.is_inference:
            assert isinstance(c.itow, dict)
            assert isinstance(c.wtoi, dict)
            if c.token_type == TOKEN_RADIX:
                # TODO: compatibility, remove in the future
                max_word_len = len(ops.number_to_base(len(c.wtoi), c.radix_base))
                c.radix_wtoi = {}
                c.radix_itow = {}
                assert c.wtoi['<PAD>'] == -1
                for k in c.wtoi:
                    if k == '<GO>':
                        idx = [c.radix_base]
                    elif k == '<EOS>':
                        idx = [c.radix_base + 1]
                    elif k == '<PAD>':
                        idx = [-1]
                    else:
                        idx = ops.number_to_base(c.wtoi[k], c.radix_base)
                        idx = [0] * (max_word_len - len(idx)) + idx
                    c.radix_wtoi[k] = idx
                    c.radix_itow['_'.join(map(str, idx))] = k
                c.radix_max_word_len = max_word_len
                assert isinstance(c.radix_wtoi, dict)
                assert isinstance(c.radix_max_word_len, int)
        else:
            fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format('itow'))
            with open(fp + '.json', 'r') as f:
                itow = c.itow = json.load(f)
            fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format('wtoi'))
            with open(fp + '.json', 'r') as f:
                wtoi = c.wtoi = json.load(f)
            
            if c.token_type == TOKEN_CHAR:
                include_symbols = 'insta' in c.dataset_file_pattern
                c.wtoi, c.itow = self.build_char_vocab(wtoi, include_symbols=include_symbols)
            elif c.token_type == TOKEN_RADIX:
                max_word_len = len(ops.number_to_base(len(wtoi), c.radix_base))
                c.radix_wtoi = {}
                c.radix_itow = {}
                assert wtoi['<PAD>'] == -1
                for k in wtoi:
                    if k == '<GO>':
                        idx = [c.radix_base]
                    elif k == '<EOS>':
                        idx = [c.radix_base + 1]
                    elif k == '<PAD>':
                        idx = [-1]
                    else:
                        idx = ops.number_to_base(wtoi[k], c.radix_base)
                        idx = [0] * (max_word_len - len(idx)) + idx
                    c.radix_wtoi[k] = idx
                    c.radix_itow['_'.join(map(str, idx))] = k
                c.radix_max_word_len = max_word_len
            else:
                assert c.token_type == TOKEN_WORD
        c.vocab_size = len(c.itow)
    
    @staticmethod
    def build_char_vocab(wtoi, include_symbols=True):
        pad_value = wtoi['<PAD>']
        char_list = list(string.digits + string.ascii_lowercase)
        
        ctoi = {}
        itoc = {}
        idx = pad_value
        ctoi['<PAD>'] = idx
        itoc[idx] = '<PAD>'
        idx += 1
        ctoi[' '] = idx
        itoc[idx] = ' '
        idx += 1
        
        for c in char_list:
            ctoi[c] = idx
            itoc[idx] = c
            idx += 1
        if include_symbols:
            ctoi['U'] = idx  # hack for emojis
            itoc[idx] = 'U'
            idx += 1
            for c in string.punctuation:
                ctoi[c] = idx
                itoc[idx] = c
                idx += 1
        ctoi['<GO>'] = len(ctoi)
        ctoi['<EOS>'] = len(ctoi)
        itoc[len(itoc)] = '<GO>'
        itoc[len(itoc)] = '<EOS>'
        itoc = dict((str(i), c) for i, c in itoc.items())
        return ctoi, itoc
    
    def _get_data(self, split):
        c = self.config
        if '{}' not in c.dataset_file_pattern:
            raise ValueError('`dataset_file_pattern` must have `{}`.')
        if self.is_inference:
            self.filenames_infer = self._get_test_set_files()
            batch_size = c.batch_size_infer
            data = []
            for f in self.filenames_infer:
                data.append([f, ['null']])
            assert len(data) % batch_size == 0
        
        else:
            # Data format: filepath,w0 w1 w2 w3 w4 ... wN
            fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format(split))
            with open(fp + '.txt', 'r') as f:
                data = [l.strip().split(',') for l in f.readlines()]
            data = [[l[0], l[1].split(' ')] for l in data]
        return data
    
    def _get_test_set_files(self):
        c = self.config
        assert c.infer_set in VALID_INFER_SET
        
        if c.infer_set in [SET_COCO_TEST, SET_COCO_VALID]:
            if c.infer_set == SET_COCO_TEST:
                coco_dir = 'test2014'
            else:
                coco_dir = 'val2014'
                c.batch_size_infer = 61
            fname_list = os.listdir(pjoin(c.dataset_dir, coco_dir))
            filenames_infer = [pjoin(c.dataset_dir, coco_dir, ff) for ff in fname_list]
        else:
            if c.infer_set == SET_TEST:
                fname_list = 'filenames_test.txt'
            else:
                fname_list = 'filenames_valid.txt'
            with open(pjoin(c.dataset_dir, 'captions', fname_list)) as f:
                filenames_infer = [l.strip() for l in f.readlines()]
        return filenames_infer
    
    def _get_eval_batch_size(self):
        c = self.config
        if c.batch_size_eval < 0:
            if 'coco' in c.dataset_file_pattern:
                c.batch_size_eval = 61
            elif 'insta' in c.dataset_file_pattern:
                c.batch_size_eval = 50
            else:
                raise ValueError(ERROR_UNKNOWN_DATASET)
        return c.batch_size_eval
    
    def _get_buckets(self):
        c = self.config
        # Calculate bucket sizes
        if c.token_type == TOKEN_CHAR:
            if 'coco' in c.dataset_file_pattern:
                buckets = [45, 55, 70]
            elif 'insta' in c.dataset_file_pattern:
                buckets = [29, 42, 61]
            else:
                raise ValueError(ERROR_UNKNOWN_DATASET)
        else:
            if 'coco' in c.dataset_file_pattern:
                buckets = [11, 13, 15]  # MSCOCO word-based
            elif 'insta' in c.dataset_file_pattern:
                buckets = [7, 10, 13]
            else:
                raise ValueError(ERROR_UNKNOWN_DATASET)
            if c.token_type == TOKEN_RADIX:
                max_word_len = len(ops.number_to_base(len(c.wtoi), c.radix_base))
                buckets = [b * max_word_len for b in buckets]
        return buckets
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        # Read data files
        data = self._get_data(split)
        data_len = len(data)
        
        buckets = self._get_buckets()
        c.split_sizes[split] = data_len
        
        is_training = 'train' in split and not self.is_inference
        if self.is_inference:
            batch_size = c.batch_size_infer
        else:
            if is_training:
                # try:
                #     gs = c.accum_grads_step
                # except:
                #     gs = 1
                gs = 1
                batch_size = c.batch_size_train
                self.config.max_step = int(data_len / batch_size * c.max_epoch / gs)
            else:
                # num_threads = 1
                batch_size = self._get_eval_batch_size()
                assert data_len % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            im_size = c.cnn_input_size
            augment = is_training and c.cnn_input_augment
            logger.debug('Augment {} images: {}'.format(split, augment))
            im_prepro_fn = prepro_factory.get_preprocessing(c.cnn_name, is_training=augment)
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: self._generator_fn(data=data, is_training=is_training),
                output_shapes=(None, [None]),
                output_types=(tf.string, tf.int32))
            # Read the images, Pre-process / Augment
            dataset = dataset.map(self.read_image, num_parallel_calls=2)
            dataset = dataset.map(lambda im, cap: (
                im_prepro_fn(im, im_size[0], im_size[1]), cap), num_parallel_calls=2)
            # Apply length bucketing and batching, Prefetch
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                element_length_func=lambda im, cap: tf.shape(cap)[0],
                bucket_boundaries=buckets,
                bucket_batch_sizes=[batch_size] * (len(buckets) + 1),
                padded_shapes=None,
                padding_values=(.0, c.wtoi['<PAD>']),
                pad_to_bucket_boundary=False))
            dataset = dataset.map(
                lambda im, cap: self.set_shape(batch_size, im, cap))
            dataset = dataset.prefetch(2)
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    def _generator_fn(self, data, is_training):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                caption = self._tokens_to_id(d[1])
                yield pjoin(c.dataset_dir, d[0]), caption
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
    
    def _tokens_to_id(self, tokens):
        c = self.config
        if self.is_inference:
            caption = [0]
        else:
            if c.token_type == TOKEN_WORD:
                caption = [c.wtoi.get(w, c.wtoi['<UNK>']) for w in tokens]
            
            elif c.token_type == TOKEN_RADIX:
                caption = [c.radix_wtoi.get(w, c.radix_wtoi['<UNK>']) for w in tokens]
                caption = list(chain.from_iterable(caption))  # flatten
            
            elif c.token_type == TOKEN_CHAR:
                caption = [c.wtoi[ch] for ch in ' '.join(tokens[1:-1])]
                caption = [c.wtoi['<GO>']] + caption + [c.wtoi['<EOS>']]
            else:
                raise ValueError('Invalid token type: {}'.format(c.token_type))
        return np.array(caption).astype(np.int32)
    
    @staticmethod
    def read_image(fp, *args):
        image = tf.image.decode_image(tf.read_file(fp), channels=3)
        image.set_shape([None, None, 3])
        # image = tf.random_normal(shape=[224, 224, 3])
        return (image,) + args
    
    @staticmethod
    def set_shape(batch_size, img, cap):
        img.set_shape([batch_size] + _shape(img)[1:])
        cap.set_shape([batch_size, None])
        return img, cap


class CaptionInputSCST(CaptionInput):
    """ Input Manager object."""
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        is_training = 'train' in split and not self.is_inference
        if not is_training:
            return None
        
        # Read data files
        data = self._get_data(split)
        data_dict = {}
        for d in data:
            if d[0] not in data_dict:
                data_dict[d[0]] = []
            s = ' '.join(d[1])
            s = s.replace('<GO> ', '').replace(' <EOS>', '')
            data_dict[d[0]].append(s)
        data = list(data_dict.items())
        data_len = len(data)
        del data_dict
        
        c.split_sizes[split] = data_len
        batch_size = c.batch_size_train
        self.config.max_step = int(data_len / batch_size * c.max_epoch)
        
        with tf.name_scope('batch_{}'.format(split)):
            im_size = c.cnn_input_size
            augment = is_training and c.cnn_input_augment
            logger.debug('Augment {} images: {}'.format(split, augment))
            im_prepro_fn = prepro_factory.get_preprocessing(c.cnn_name, is_training=augment)
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: self._generator_fn(data=data, is_training=is_training),
                output_shapes=(None, [None]),
                output_types=(tf.string, tf.string))
            # Read the images
            dataset = dataset.map(self.read_image, num_parallel_calls=2)
            # Pre-process / Augment the images
            dataset = dataset.map(lambda im, cap: (
                im_prepro_fn(im, im_size[0], im_size[1]), cap), num_parallel_calls=2)
            # Apply batching, Prefetch
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
            dataset = dataset.prefetch(5)
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    def _generator_fn(self, data, is_training):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of captions for that image
                # for MSCOCO Karpathy split, there are 113,287 train images
                #     308 (0.272 %) have more than 5 GT captions
                #     for convenience, we just take 5 captions
                yield pjoin(c.dataset_dir, d[0]), np.array(d[1][:5])
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
    
    def captions_to_batched_ids(self, hypos):
        """
        Generates batched IDs with padding for SCST training.
        Used as GT for XE objective.
        """
        c = self.config
        assert c.token_type in VALID_TOKEN_TYPE
        
        hypos_idx = []
        for h in hypos:
            if c.token_type == TOKEN_RADIX:
                h = ['<GO>'] + h[0].split() + ['<EOS>']
                h = [c.radix_wtoi.get(w, c.radix_wtoi['<UNK>']) for w in h]
                h = np.concatenate(h)
            elif c.token_type == TOKEN_WORD:
                h = ['<GO>'] + h[0].split() + ['<EOS>']
                h = [c.wtoi.get(w, c.wtoi['<UNK>']) for w in h]
                h = np.array(h)
            elif c.token_type == TOKEN_CHAR:
                h = [c.wtoi[ch] for ch in h[0]]
                h = [c.wtoi['<GO>']] + h + [c.wtoi['<EOS>']]
                h = np.array(h)
            hypos_idx.append(h)
        
        assert len(hypos_idx[0].shape) == 1
        max_hypo_len = max([hy.shape[0] for hy in hypos_idx])
        assert max_hypo_len > 1
        for i, h in enumerate(hypos_idx):
            h = np.pad(h, pad_width=[0, max_hypo_len - len(h)],
                       mode='constant', constant_values=c.wtoi['<PAD>'])
            hypos_idx[i] = h
        hypos_idx = np.stack(hypos_idx, axis=0)
        return hypos_idx


class CaptionInputAE(CaptionInput):
    """ Input Manager object for autoencoder."""
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        # Read data files
        data = self._get_data(split)
        data_len = len(data)
        
        buckets = self._get_buckets()
        c.split_sizes[split] = data_len
        
        is_training = 'train' in split and not self.is_inference
        if self.is_inference:
            batch_size = c.batch_size_infer
        else:
            if is_training:
                gs = 1
                batch_size = c.batch_size_train
                self.config.max_step = int(data_len / batch_size * c.max_epoch / gs)
            else:
                # num_threads = 1
                batch_size = self._get_eval_batch_size()
                assert data_len % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: self._generator_fn(data=data, is_training=is_training),
                output_shapes=([None], [None]),
                output_types=(tf.int32, tf.int32))
            # Apply length bucketing and batching, Prefetch
            _PAD = '<PAD>'
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                element_length_func=lambda cap_rev, cap: tf.shape(cap)[0],
                bucket_boundaries=buckets,
                bucket_batch_sizes=[batch_size] * (len(buckets) + 1),
                padded_shapes=None,
                padding_values=(c.wtoi[_PAD], c.wtoi[_PAD]),
                pad_to_bucket_boundary=False))
            dataset = dataset.map(
                lambda cap_rev, cap: self.set_shape(batch_size, cap_rev, cap))
            dataset = dataset.prefetch(2)
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    def _generator_fn(self, data, is_training):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                caption = self._tokens_to_id(d[1])
                caption_rev = caption.copy()[1:-1][::-1]  # remove <GO> and <EOS> tokens, reverse
                yield caption_rev, caption
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
    
    @staticmethod
    def set_shape(batch_size, cap_rev, cap):
        cap_rev.set_shape([batch_size, None])
        cap.set_shape([batch_size, None])
        return cap_rev, cap


class CaptionInputVAEX(CaptionInputAE):
    """ Input Manager object for variational autoencoder."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.config
        _non_attr_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'at', 'are', 'some',
                           'it', 'by', 'has', 'there', 'for', 'from', 'its', 'their', 'his', 'her',
                           'this', 'that', 'very', 'so']
        if c.token_type == TOKEN_WORD:
            num_attr = 3000
            attributes = []
            for idx, word in sorted(c.itow.items(), key=lambda x: int(x[0])):
                if word not in _non_attr_words and len(attributes) < num_attr and '<' not in word:
                    attributes.append(word)
            # words = [v for k, v in sorted(c.itow.items(), key=lambda x: int(x[0])) if v not in attributes]
            # c.itow = {}
            # c.wtoi = {}
            # for i, w in enumerate(words):
            #     c.itow[i - 1] = w
            #     c.wtoi[w] = i - 1
            # c.itow = {k: v for k, v in c.itow.items() if v not in attributes}
            # c.wtoi = {k: v for k, v in c.wtoi.items() if k not in attributes}
            # c.attributes = np.array(attributes).astype(np.int32)
            # print(attributes)
            # print([(k, c.itow[k]) for k in sorted(c.itow.keys(), key=lambda x: int(x))])
            c.attributes = attributes
            # c.vocab_size = len(c.itow)
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        # Read data files
        data = self._get_data(split)
        data_len = len(data)
        
        buckets = self._get_buckets()
        c.split_sizes[split] = data_len
        
        is_training = 'train' in split and not self.is_inference
        if self.is_inference:
            batch_size = c.batch_size_infer
            assert data_len % batch_size == 0
        else:
            if is_training:
                gs = 1
                batch_size = c.batch_size_train
                self.config.max_step = int(data_len / batch_size * c.max_epoch / gs)
            else:
                # num_threads = 1
                batch_size = self._get_eval_batch_size()
                assert data_len % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: self._generator_fn(data=data, is_training=is_training),
                output_shapes=([None], [None], [None]),
                output_types=(tf.int32, tf.int32, tf.int32))
            
            _PAD = '<PAD>'
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                element_length_func=lambda cap_rev, cap, attr: tf.shape(cap)[0],
                bucket_boundaries=buckets,
                bucket_batch_sizes=[batch_size] * (len(buckets) + 1),
                padded_shapes=None,
                padding_values=(c.wtoi[_PAD], c.wtoi[_PAD], c.wtoi[_PAD]),
                pad_to_bucket_boundary=False))
            dataset = dataset.map(
                lambda cap_rev, cap, attr: self.set_shape(batch_size, cap_rev, cap, attr))
            dataset = dataset.prefetch(2)
            
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    def _generator_fn(self, data, is_training):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                attributes = []
                skeleton = []
                for _ in d[1]:
                    if _ in c.attributes:
                        attributes.append(_)
                        skeleton.append('<UNK>')
                    else:
                        skeleton.append(_)
                # print(skeleton)
                if len(attributes) == 0:
                    print('!!!')
                    attributes = ['<UNK>']
                attributes = self._tokens_to_id(attributes)
                skeleton = self._tokens_to_id(skeleton)[1:-1][::-1]
                caption = self._tokens_to_id(d[1])
                yield skeleton, caption, attributes
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
    
    @staticmethod
    def set_shape(batch_size, cap_rev, cap, attr):
        cap_rev.set_shape([batch_size, None])
        cap.set_shape([batch_size, None])
        attr.set_shape([batch_size, None])
        return cap_rev, cap, attr
