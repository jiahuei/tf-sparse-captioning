# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""
import os
import pickle
from time import localtime, strftime
from common.natural_sort import natural_keys


VOCAB_DICT = ['wtoi', 'itow', 'ctoi', 'itoc', 'radix_wtoi', 'radix_itow']


class Config(object):
    """ Configuration object."""
    
    def __init__(self, **kwargs):
        for key, value in sorted(kwargs.items()):
            setattr(self, key, value)

    # noinspection PyUnresolvedReferences
    def save_config_to_file(self):
        params = vars(self)
        keys = sorted(params.keys(), key=natural_keys)
        txt_dump = ['%s = %s' % (k, params[k]) for k in keys if k not in VOCAB_DICT]
        config_name = 'config___%s.txt' % strftime('%Y-%m-%d_%H-%M-%S', localtime())
        with open(os.path.join(self.log_path, config_name), 'w') as f:
            f.write('\r\n'.join(txt_dump))
        # Save the dictionary instead of the object for maximum flexibility
        # Avoid this error:
        # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
        with open(os.path.join(self.log_path, 'config.pkl'), 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # noinspection PyUnresolvedReferences
    def overwrite_safety_check(self, overwrite):
        """ Exits if log_path exists but `overwrite` is set to `False`."""
        path_exists = os.path.exists(self.log_path)
        if path_exists:
            if not overwrite:
                print('\nINFO: log_path already exists. '
                      'Set `overwrite` to True? Exiting now.')
                raise SystemExit
            else:
                print('\nINFO: log_path already exists. '
                      'The directory will be overwritten.')
        else:
            print('\nINFO: log_path does not exist. '
                  'The directory will be created.')
            os.makedirs(self.log_path)


def load_config(config_filepath):
    with open(config_filepath, 'rb') as f:
        c_dict = pickle.load(f)
    config = Config(**c_dict)
    return config
