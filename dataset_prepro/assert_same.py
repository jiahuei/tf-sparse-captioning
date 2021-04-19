# -*- coding: utf-8 -*-
"""
Created on 24 Dec 2019 20:40:06

@author: jiahuei
"""


def get_dict(fp):
    data = {}
    with open(fp, 'r') as f:
        for ll in f.readlines():
            _ = ll.split(',')
            data[_[0]] = _[1].rstrip()
    return data


def dump(data, keys, out_path):
    out_str = ''
    for k in keys:
        out_str += '{},{}\r\n'.format(k, data[k])
    with open(out_path, 'w') as f:
        f.write(out_str)


SPLITS = ['train', 'valid', 'test']

for split in SPLITS:
    print('Checking {} ... '.format(split), end='')
    a = get_dict('/master/datasets/mscoco/captions_py2/mscoco_{}_v25595_s15.txt'.format(split))
    b = get_dict('/master/datasets/mscoco/captions/mscoco_{}_v25595_s15.txt'.format(split))
    
    a_keys = sorted(a.keys())
    b_keys = sorted(b.keys())
    
    a_values = sorted(a.values())
    b_values = sorted(b.values())
    if a_keys == b_keys and a_values == b_values:
        print('OK')
    else:
        print('DIFFERENT')
    del a, b

# dump(a, a_keys, '/master/datasets/insta/py2.txt')
# dump(b, a_keys, '/master/datasets/insta/py3.txt')

