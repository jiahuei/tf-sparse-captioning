# -*- coding: utf-8 -*-
"""
Created on 31 Mar 2020 23:36:49

@author: jiahuei
"""
import os
import json
import logging
import random
import h5py
import tensorflow as tf
import numpy as np
from itertools import chain
from common.nets import nets_factory
from common.inputs import image_caption as cap
from common.inputs.preprocessing import preprocessing_factory as prepro_factory
from common import ops_v1 as ops

logger = logging.getLogger(__name__)
_shape = ops.shape
pjoin = os.path.join
slim = tf.contrib.slim


class VQAInput:
    """ Input Manager object."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """
        if is_inference:
            raise NotImplementedError('Please use `VQAInference` for inference.')
        self.is_inference = is_inference
        logger.debug('Using `common.inputs.vqa_v2` (v0).')
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
        random.seed(c.rand_seed)
        
        # Read vocab files
        assert c.token_type in cap.VALID_TOKEN_TYPE
        self._set_vocab()
        
        # Setup input pipelines
        self._get_data()
        self.batch_train = self._batch_setup('train')
        if c.dataset_combine_val:
            self.batch_eval = None
        else:
            self.batch_eval = self._batch_setup('valid')
        logger.info('Input pipelines setup complete.')
    
    def _set_vocab(self):
        c = self.config
        
        if self.is_inference:
            assert isinstance(c.itow, dict)
            assert isinstance(c.wtoi, dict)
            assert isinstance(c.answer_list, list)
            if c.token_type == cap.TOKEN_RADIX:
                assert isinstance(c.radix_wtoi, dict)
                assert isinstance(c.radix_itow, dict)
                assert isinstance(c.radix_max_word_len, int)
        else:
            # vqa_v2_ac9_q14_data, vqa_v2_ac9_q14_qwc5_vocab
            # dataset_file_prefix = c.dataset_file_prefix.split('qwc')[0]
            if not c.dataset_file_prefix.endswith('_'):
                c.dataset_file_prefix += '_'
            fp = pjoin(c.dataset_dir, 'vqa_v2', c.dataset_file_prefix + 'vocab.json')
            with open(fp, 'r') as f:
                vocab = json.load(f)
            wtoi = c.wtoi = vocab['wtoi']
            itow = c.itow = vocab['itow']
            c.answer_list = vocab['answer_list']
            
            if c.token_type == cap.TOKEN_CHAR:
                c.wtoi, c.itow = cap.CaptionInput.build_char_vocab(wtoi, include_symbols=True)
            elif c.token_type == cap.TOKEN_RADIX:
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
                assert c.token_type == cap.TOKEN_WORD
        c.vocab_size = len(c.itow)
    
    def _get_data(self):
        c = self.config
        # vqa_v2_ac9_q14_data, vqa_v2_ac9_q14_qwc5_vocab
        assert '_qwc' in c.dataset_file_prefix, '`dataset_file_prefix` must contain `_qwc`.'
        dataset_file_prefix = c.dataset_file_prefix.split('qwc')[0]
        # Data format: filepath,w0 w1 w2 w3 w4 ... wN
        fp = pjoin(c.dataset_dir, 'vqa_v2', dataset_file_prefix + 'data.json')
        with open(fp, 'r') as f:
            data = json.load(f)
        if c.dataset_combine_val:
            data['train'] = data['train'] + data['valid']
            del data['valid']
        self.data = data
        c.question_max_len = int(c.dataset_file_prefix.split('_q')[1].split('_')[0])
        if c.token_type == cap.TOKEN_RADIX:
            c.question_max_len *= c.radix_max_word_len
        if c.use_image_features == 'fixed_36':
            fp = pjoin(c.dataset_dir, 'vqa_v2', 'trainval_36_features.h5')
            self.features = h5py.File(fp, 'r')['image_features']
    
    def _get_eval_batch_size(self):
        c = self.config
        if c.batch_size_eval < 0:
            if 'vqa_v2' in c.dataset_file_prefix:
                c.batch_size_eval = 61
            else:
                raise ValueError(cap.ERROR_UNKNOWN_DATASET)
        return c.batch_size_eval
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        # Read data files
        data_len = len(self.data[split])
        ans_softmax_size = len(c.answer_list)
        c.split_sizes[split] = data_len
        
        assert not self.is_inference
        is_training = 'train' in split
        if is_training:
            batch_size = c.batch_size_train
            self.config.max_step = int(data_len / batch_size * c.max_epoch)
        else:
            # num_threads = 1
            batch_size = self._get_eval_batch_size()
            assert data_len % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            if c.use_image_features is None:
                im_size = c.cnn_input_size
                augment = is_training and c.cnn_input_augment
                logger.debug('Augment {} images: {}'.format(split, augment))
                im_prepro_fn = prepro_factory.get_preprocessing(c.cnn_name, is_training=augment)
                # Fetch filepaths and captions from data generator
                dataset = tf.data.Dataset.from_generator(
                    generator=lambda: self._generator_fn(data=self.data[split], is_training=is_training),
                    output_shapes=([], [None], [], [ans_softmax_size]),
                    output_types=(tf.string, tf.int32, tf.int64, tf.float32))
                # Read the images, Pre-process / Augment
                dataset = dataset.map(cap.CaptionInput.read_image, num_parallel_calls=2)
                # dataset = dataset.map(self._fake_image_read, num_parallel_calls=2)
                dataset = dataset.map(lambda im, *args: (
                        (im_prepro_fn(im, im_size[0], im_size[1]),) + args), num_parallel_calls=2)
            else:
                dataset = tf.data.Dataset.from_generator(
                    generator=lambda: self._generator_fn(data=self.data[split], is_training=False),
                    output_shapes=([36, 2048], [None], [], [ans_softmax_size]),
                    output_types=(tf.float32, tf.int32, tf.int64, tf.float32))
            # Apply batching, Prefetch
            # dataset = dataset.padded_batch(
            #     batch_size=batch_size,
            #     padded_shapes=([im_size[0], im_size[1], 3],
            #                    c.question_max_len,
            #                    [],
            #                    ans_softmax_size),
            #     padding_values=(0., c.wtoi['<PAD>'], np.array(0, dtype=np.int64), 0.))
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(lambda *args: self._set_shape(batch_size, *args))
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
                # d is a dict
                question = self._tokens_to_id(d['question_tokens'])
                print(d['question_tokens'])
                print(question)
                print(d['answers'])
                print(d['GT_multi_label'])
                answer_array = np.zeros(shape=(len(c.answer_list),), dtype=np.float32)
                for lbl_id, lbl_val in d['GT_multi_label']:
                    answer_array[lbl_id] = lbl_val
                print(answer_array[lbl_id])
                if c.use_image_features is None:
                    yield pjoin(c.dataset_dir, d['image_path']), question, d['question_id'], answer_array
                else:
                    feat_idx = self.data['imgid_to_idx'][str(d['image_id'])]
                    yield self.features[feat_idx, :, :], question, d['question_id'], answer_array
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
    
    def _tokens_to_id(self, tokens):
        c = self.config
        if c.token_type == cap.TOKEN_WORD:
            caption = [c.wtoi.get(w, c.wtoi['<UNK>']) for w in tokens]
        
        elif c.token_type == cap.TOKEN_RADIX:
            caption = [c.radix_wtoi.get(w, c.radix_wtoi['<UNK>']) for w in tokens]
            caption = list(chain.from_iterable(caption))  # flatten
        
        elif c.token_type == cap.TOKEN_CHAR:
            caption = [c.wtoi[ch] for ch in ' '.join(tokens)]
            raise NotImplementedError
        else:
            raise ValueError('Invalid token type: {}'.format(c.token_type))
        # Note here we pad in front of the sentence
        paddings = [c.wtoi['<PAD>']] * (c.question_max_len - len(caption))
        caption = paddings + caption
        return np.array(caption).astype(np.int32)
    
    # @staticmethod
    # def _fake_image_read(fp, *args):
    #     image = tf.random_normal(shape=[224, 224, 3])
    #     return (image,) + args
    
    def _set_shape(self, batch_size, img, que, qid, ans=None):
        c = self.config
        img.set_shape([batch_size] + _shape(img)[1:])
        que.set_shape([batch_size, c.question_max_len])
        qid.set_shape([batch_size])
        if ans is not None:
            ans.set_shape([batch_size, len(c.answer_list)])
            return img, que, qid, ans
        else:
            return img, que, qid


# noinspection PyMissingConstructor
class VQAInference(VQAInput):
    def __init__(self, config, is_inference=True):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """
        if not is_inference:
            raise NotImplementedError('Please use `VQAInput` for training / eval.')
        self.is_inference = is_inference
        logger.debug('Using `common.inputs.vqa_v2` (v0).')
        self.config = c = config
        
        # Add new info to config
        c.split_sizes = {}
        random.seed(c.rand_seed)
        
        # Read vocab files
        assert c.token_type in cap.VALID_TOKEN_TYPE
        self._set_vocab()
        
        # Setup input pipelines
        self._get_data()
        self.batch_test = self._batch_setup('test')
        self.batch_test_dev = self._batch_setup('test-dev')
        logger.info('Input pipelines setup complete.')
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        assert self.is_inference
        c = self.config
        
        # Read data files
        data_len = len(self.data[split])
        # self.filenames_infer = [_['image_path'] for _ in self.data[split]]
        c.split_sizes[split] = data_len
        batch_size = c.batch_size_infer
        assert data_len % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            if c.use_image_features is None:
                im_size = c.cnn_input_size
                im_prepro_fn = prepro_factory.get_preprocessing(c.cnn_name, is_training=False)
                # Fetch filepaths and captions from data generator
                dataset = tf.data.Dataset.from_generator(
                    generator=lambda: self._generator_fn(data=self.data[split], is_training=False),
                    output_shapes=([], [None], []),
                    output_types=(tf.string, tf.int32, tf.int64))
                # Read the images, Pre-process / Augment
                dataset = dataset.map(cap.CaptionInput.read_image, num_parallel_calls=2)
                dataset = dataset.map(lambda im, que, qid: (
                    im_prepro_fn(im, im_size[0], im_size[1]), que, qid), num_parallel_calls=2)
            else:
                dataset = tf.data.Dataset.from_generator(
                    generator=lambda: self._generator_fn(data=self.data[split], is_training=False),
                    output_shapes=([36, 2048], [None], []),
                    output_types=(tf.float32, tf.int32, tf.int64))
            # Apply batching, Prefetch
            # dataset = dataset.padded_batch(
            #     batch_size=batch_size,
            #     padded_shapes=([im_size[0], im_size[1], 3], [c.question_max_len], []),
            #     padding_values=(0., c.wtoi['<PAD>'], np.array(0, dtype=np.int64)))
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(lambda *args: self._set_shape(batch_size, *args))
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
        assert not is_training
        c = self.config
        for d in data:
            # d is a dict
            question = self._tokens_to_id(d['question_tokens'])
            if c.use_image_features is None:
                yield pjoin(c.dataset_dir, d['image_path']), question, d['question_id']
            else:
                feat_idx = self.data['imgid_to_idx'][str(d['image_id'])]
                yield self.features[feat_idx, :, :], question, d['question_id']
