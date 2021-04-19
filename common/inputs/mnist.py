# -*- coding: utf-8 -*-
"""
Created on 31 Mar 2020 23:34:30

@author: jiahuei
"""
from common.inputs.mnist_utils import read_data as read_mnist
import tensorflow as tf
import random
import logging

logger = logging.getLogger(__name__)


class MNISTInput(object):
    """ Input Manager object."""

    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """
        logger.debug('Using `common.inputs.mnist` (v1).')
        del is_inference
        # Add new info to config
        config.split_sizes = {}
        self.config = c = config
        random.seed(c.rand_seed)

        # Setup input pipelines
        mnist_data = read_mnist(c.dataset_dir)
        self._split_dataset(mnist_data, 'train', 'valid', int(mnist_data['train']['input'].shape[0] * 0.1))
        self.batch_train = self._batch_setup('train', mnist_data['train'])
        self.batch_valid = self._batch_setup('valid', mnist_data['valid'])
        self.batch_test = self._batch_setup('test', mnist_data['test'])
        logger.debug('Input pipelines setup complete.')

    @staticmethod
    def _split_dataset(data_dict, source, target, number):
        """
        From https://github.com/namhoonlee/snip-public
        """
        keys = ['input', 'label']
        indices = list(range(data_dict[source]['input'].shape[0]))
        random.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        data_dict[target] = {k: data_dict[source][k][ind_target] for k in keys}
        data_dict[source] = {k: data_dict[source][k][ind_remain] for k in keys}

    def _batch_setup(self, split, data):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config

        is_training = 'train' in split
        self.config.split_sizes[split] = len(data['input'])

        if is_training:
            batch_size = c.batch_size_train
            self.config.max_step = int(len(data['input']) / batch_size * c.max_epoch)
        else:
            # num_threads = 1
            batch_size = c.batch_size_eval
            assert len(data['input']) % batch_size == 0

        with tf.name_scope('batch_{}'.format(split)):
            dataset = tf.data.Dataset.from_generator(
                generator=lambda: self._generator_fn(data, is_training),
                output_shapes=([28, 28], [1]),
                output_types=(tf.float32, tf.int32))
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            if is_training:
                dataset = dataset.prefetch(2)

            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch

    # noinspection PyMethodMayBeStatic
    def _generator_fn(self, data, is_training=True):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        # c = self.config
        data_zipped = zip(data['input'], data['label'])
        del data

        idx = 0
        if is_training:
            random.shuffle(data_zipped)
            logger.debug('Training data shuffled, idx {:3,d}'.format(idx))

        while True:
            for d in data_zipped:
                yield (d[0][:, :, 0], [d[1]])
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data_zipped)
                logger.debug('Training data shuffled, idx {:3,d}'.format(idx))
