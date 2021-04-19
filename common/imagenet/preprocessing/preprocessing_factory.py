# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Based on
https://github.com/tensorflow/models/tree/6766e6ddfabefbd1003d5275fa5cc32f18196c11/research/slim/preprocessing

Changes include:
    - update to TF 2.x symbols

Contains a factory for building various models.

Original mapping:

    preprocessing_fn_map = {
        'cifarnet': cifarnet_preprocessing,
        'inception': inception_preprocessing,
        'inception_v1': inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'lenet': lenet_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'mobilenet_v2': inception_preprocessing,
        'mobilenet_v2_035': inception_preprocessing,
        'mobilenet_v3_small': inception_preprocessing,
        'mobilenet_v3_large': inception_preprocessing,
        'mobilenet_v3_small_minimalistic': inception_preprocessing,
        'mobilenet_v3_large_minimalistic': inception_preprocessing,
        'mobilenet_edgetpu': inception_preprocessing,
        'mobilenet_edgetpu_075': inception_preprocessing,
        'mobilenet_v2_140': inception_preprocessing,
        'nasnet_mobile': inception_preprocessing,
        'nasnet_large': inception_preprocessing,
        'pnasnet_mobile': inception_preprocessing,
        'pnasnet_large': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'resnet_v1_200': vgg_preprocessing,
        'resnet_v2_50': vgg_preprocessing,
        'resnet_v2_101': vgg_preprocessing,
        'resnet_v2_152': vgg_preprocessing,
        'resnet_v2_200': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
    }
"""

from common.imagenet.preprocessing import inception_preprocessing
from common.imagenet.preprocessing import vgg_preprocessing


def get_preprocessing(name, is_training=False, use_grayscale=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).
    
    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      use_grayscale: Whether to convert the image from RGB to grayscale. Ignored for EfficientNet.
    
    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).
    
    Raises:
      ValueError: If Preprocessing `name` is not recognized.
    """
    assert isinstance(name, str)
    # if name.startswith('efficientnet_b'):
    #     def preprocessing_fn(image_bytes, output_height, output_width, **kwargs):
    #         assert output_height == output_width
    #         return efficientnet_preprocessing.preprocess_image(
    #             image_bytes,
    #             is_training=is_training,
    #             image_size=output_height,
    #             **kwargs)
    #
    #     return preprocessing_fn
    if name.startswith(('inception', 'resnet_v2', 'mobilenet', 'nasnet', 'pnasnet')):
        # https://github.com/tensorflow/models/tree/v1.12.0/research/slim
        prepro = inception_preprocessing
    elif name.startswith(('resnet_v1', 'vgg', 'densenet')):
        prepro = vgg_preprocessing
    # elif name in ['lenet']:
    #     prepro = lenet_preprocessing
    # elif name in ['cifarnet']:
    #     prepro = cifarnet_preprocessing
    else:
        raise ValueError('Invalid CNN / preprocessing choice: `{}`'.format(name))
    
    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return prepro.preprocess_image(
            image,
            output_height,
            output_width,
            is_training=is_training,
            use_grayscale=use_grayscale,
            **kwargs)
    
    return preprocessing_fn
