# -*- coding: utf-8 -*-
"""
Created on 02 Mar 2020 22:34:49

@author: jiahuei

Based on:
https://github.com/tensorflow/models/blob/v1.13.0/research/slim/datasets/imagenet.py
"""
import os
import urllib.request
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from common.nets import nets_factory
from common.imagenet.preprocessing.preprocessing_factory import get_preprocessing

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_CLASSES = 1001


class CNNModel:
    def __init__(self, cnn_name, images, labels):
        self.imgs = images
        self.labels = labels
        cnn_fn = nets_factory.get_network_fn(
            name=cnn_name,
            num_classes=1000 if cnn_name.lower().startswith(('resnet', 'vgg')) else _NUM_CLASSES,
            weight_decay=0.0,
            is_training=False,
        )
        try:
            self.probabilities, end_points = cnn_fn(self.imgs, create_aux_logits=False)
        except TypeError:
            self.probabilities, end_points = cnn_fn(self.imgs)
        if self.probabilities.shape[-1] == 1001:
            self.probabilities = self.probabilities[:, 1:]
        self.top1 = tf.nn.in_top_k(targets=self.labels, predictions=self.probabilities, k=1)
        self.top5 = tf.nn.in_top_k(targets=self.labels, predictions=self.probabilities, k=5)
        self.saver = tf.train.Saver()

    def restore_weights(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)

    def evaluate(self, session):
        probabilities, top1, top5 = session.run([self.probabilities, self.top1, self.top5])
        return probabilities, top1, top5


def retrieve_and_read(url):
    basename = url.split('/')[-1]
    assert basename.endswith('.txt')
    filename = os.path.join(CURR_DIR, basename)
    if not os.path.isfile(filename):
        try:
            filename, _ = urllib.request.urlretrieve(url, filename=filename)
        except FileNotFoundError:
            filename, _ = urllib.request.urlretrieve(url)
    with open(filename, 'r') as f:
        data = [s.strip() for s in f.readlines()]
    return data


def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string.
    Returns:
        labels_to_names: dictionary where keys are integers from 0 to 1000
        and values are human-readable names.
    We retrieve a synset file, which contains a list of valid synset labels used
    by ILSVRC competition. There is one synset one per line, eg.
            #   n01440764
            #   n01443537
    We also retrieve a synset_to_human_file, which contains a mapping from synsets
    to human-readable names for every synset in Imagenet. These are stored in a
    tsv format, as follows:
            #   n02119247    black fox
            #   n02119359    silver fox
    We assign each synset (in alphabetical order) an integer, starting from 1
    (since 0 is reserved for the background class).
    Code is based on
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
    """

    # pylint: disable=g-line-too-long
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

    synset_list = retrieve_and_read(synset_url)
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    synset_to_human_list = retrieve_and_read(synset_to_human_url)
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def create_image_name_to_val_label_mapping(valid_dir):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
    label_url = '{}/imagenet_2012_validation_synset_labels.txt'.format(base_url)
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    labels = retrieve_and_read(label_url)
    synsets = retrieve_and_read(synset_url)
    assert len(labels) == _SPLITS_TO_SIZES['validation']
    assert len(synsets) == (_NUM_CLASSES - 1)

    synset_to_int = {syn: i for i, syn in enumerate(synsets)}

    image_paths = sorted(os.listdir(valid_dir))
    assert len(image_paths) == _SPLITS_TO_SIZES['validation']

    img_to_label = {img: synset_to_int[lbl] for img, lbl in zip(image_paths, labels)}
    image_paths = [os.path.join(valid_dir, _) for _ in image_paths]
    img_lbl = list(zip(image_paths, [str(synset_to_int[lbl]) for lbl in labels]))
    return img_to_label, img_lbl


def create_dataset(data, cnn_name, image_size, batch_size, is_training=False):
    def _preprocessing_fn(fpath):
        # Read image file
        image = tf.image.decode_image(tf.read_file(fpath), channels=3)
        image.set_shape([None, None, 3])
        _fn = get_preprocessing(cnn_name, is_training=is_training, use_grayscale=False)
        return _fn(image, image_size[0], image_size[1])

    dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_training:
        buffer = min(len(data), 5000)
        dataset = dataset.shuffle(
            buffer_size=buffer, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.repeat(count=None)
    else:
        dataset = dataset.repeat(count=1)
    # Read and augment image
    dataset = dataset.map(lambda x: (_preprocessing_fn(x[0]), tf.string_to_number(x[1], tf.int64)),
                          num_parallel_calls=2)
    # Perform batching, prefetch
    # For training, all batch size will be equal as `repeat(count=None)`
    # For evaluation, batch sizes might differ across batches.
    # `drop_remainder=False` ensures that all eval data will be served.
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(5)
    # Get the dataset iterator
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


def eval_validation(valid_dir, cnn_name, checkpoint_path, batch_size, visualise=False):
    # def_size = get_cnn_default_input_size(cnn_name, is_training=False)
    def_size = (224, 224)
    _, img_lbl = create_image_name_to_val_label_mapping(valid_dir)
    # img_lbl = img_lbl[:100]
    assert len(img_lbl) % batch_size == 0

    g = tf.Graph()
    with g.as_default():
        img, lbl = create_dataset(
            img_lbl, cnn_name, image_size=def_size, batch_size=batch_size)
        model = CNNModel(cnn_name, img, lbl)
    sess = tf.Session(graph=g)
    with sess:
        # ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
        # ckpt_vars_namelist = set(ckpt_reader.get_variable_to_shape_map().keys())
        # print(ckpt_vars_namelist)
        model.restore_weights(sess, checkpoint_path)
        g.finalize()
        all_top1 = []
        all_top5 = []
        probs = None
        total_steps = int(len(img_lbl) / batch_size)
        for s in tqdm(range(total_steps), desc='Evaluating CNN ({} @ {})'.format(cnn_name, def_size[0])):
            probs, top1, top5 = model.evaluate(sess)
            assert len(probs.shape) == 2
            all_top1.append(top1)
            all_top5.append(top5)
            # print(all_top5)
            # break
    if visualise:
        id2label = create_readable_names_for_imagenet_labels()
        preds = [id2label[i + 1] for i in np.argmax(probs, axis=1)]
        print(preds)
    del g
    all_top1 = np.concatenate(all_top1)
    all_top5 = np.concatenate(all_top5)
    assert len(all_top1.shape) == 1
    assert len(all_top5.shape) == 1
    acc_top1 = float(np.sum(all_top1) / all_top1.shape[0])
    acc_top5 = float(np.sum(all_top5) / all_top5.shape[0])
    return acc_top1, acc_top5
