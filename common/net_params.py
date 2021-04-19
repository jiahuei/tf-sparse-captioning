# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:21:36 2019

@author: jiahuei

Network parameters, preprocessing functions, etc.

"""
import os

pjoin = os.path.join

all_net_params = dict(
    vgg_16=dict(
        name='vgg_16',
        ckpt_name='vgg_16.ckpt',
        url='http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
    ),
    resnet_v1_50=dict(
        name='resnet_v1_50',
        ckpt_name='resnet_v1_50.ckpt',
        url='http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
    ),
    resnet_v1_101=dict(
        name='resnet_v1_101',
        ckpt_name='resnet_v1_101.ckpt',
        url='http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
    ),
    resnet_v1_152=dict(
        name='resnet_v1_152',
        ckpt_name='resnet_v1_152.ckpt',
        url='http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
    ),
    resnet_v2_50=dict(
        name='resnet_v2_50',
        ckpt_name='resnet_v2_50.ckpt',
        url='http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
    ),
    resnet_v2_101=dict(
        name='resnet_v2_101',
        ckpt_name='resnet_v2_101.ckpt',
        url='http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
    ),
    resnet_v2_152=dict(
        name='resnet_v2_152',
        ckpt_name='resnet_v2_152.ckpt',
        url='http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
    ),
    inception_v1=dict(
        name='inception_v1',
        ckpt_name='inception_v1.ckpt',
        url='http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
    ),
    inception_v2=dict(
        name='inception_v2',
        ckpt_name='inception_v2.ckpt',
        url='http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
    ),
    inception_v3=dict(
        name='inception_v3',
        ckpt_name='inception_v3.ckpt',
        url='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
    ),
    inception_v4=dict(
        name='inception_v4',
        ckpt_name='inception_v4.ckpt',
        url='http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
    ),
    inception_resnet_v2=dict(
        name='inception_resnet_v2',
        ckpt_name='inception_resnet_v2_2016_08_30.ckpt',
        url='http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
    ),
    mobilenet_v1=dict(
        name='mobilenet_v1',
        ckpt_name='mobilenet_v1_1.0_224.ckpt',
        url='http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz',
    ),
    mobilenet_v1_075=dict(
        name='mobilenet_v1_075',
        ckpt_name='mobilenet_v1_0.75_224.ckpt',
        url='http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224.tgz',
    ),
    mobilenet_v1_050=dict(
        name='mobilenet_v1_050',
        ckpt_name='mobilenet_v1_0.5_224.ckpt',
        url='http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz',
    ),
    mobilenet_v1_025=dict(
        name='mobilenet_v1_025',
        ckpt_name='mobilenet_v1_0.25_224.ckpt',
        url='http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz',
    ),
    mobilenet_v2=dict(
        name='mobilenet_v2',
        ckpt_name='mobilenet_v2_1.0_224.ckpt',
        url='https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
    ),
    mobilenet_v2_140=dict(
        name='mobilenet_v2_140',
        ckpt_name='mobilenet_v2_1.4_224.ckpt',
        url='https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
    ),
)
all_net_params['masked_inception_v1'] = all_net_params['inception_v1']
all_net_params['masked_mobilenet_v1'] = all_net_params['mobilenet_v1']
all_net_params['masked_mobilenet_v1_075'] = all_net_params['mobilenet_v1_075']
all_net_params['masked_mobilenet_v1_050'] = all_net_params['mobilenet_v1_050']
all_net_params['masked_mobilenet_v1_025'] = all_net_params['mobilenet_v1_025']


def get_net_params(net_name, ckpt_dir_or_file=''):
    net_params = all_net_params[net_name]
    ckpt_name = net_params['ckpt_name']
    
    if ckpt_dir_or_file is None or ckpt_dir_or_file == '':
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ckpt_dir_or_file = pjoin(base_dir, 'pretrained', ckpt_name)
    else:
        if os.path.isdir(ckpt_dir_or_file):
            ckpt_dir_or_file = pjoin(ckpt_dir_or_file, ckpt_name)
        if os.path.isfile(ckpt_dir_or_file):
            assert os.path.basename(ckpt_dir_or_file) == ckpt_name
    net_params['ckpt_path'] = ckpt_dir_or_file
    return net_params
