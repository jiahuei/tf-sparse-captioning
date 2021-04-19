# -*- coding: utf-8 -*-
"""
Created on 06 Feb 2020 23:03:28

@author: jiahuei (snipe)
"""
import os
import sys
import platform

up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
COMMON = up_dir(CURR_DIR)
BASE_DIR = up_dir(COMMON)
sys.path.insert(1, BASE_DIR)
sys.path.insert(1, COMMON)
from pprint import pprint
from common.nets import nets_factory

nets = nets_factory.networks_map.keys()
size = {}
for n in nets:
    net = nets_factory.networks_map[n]
    if hasattr(net, 'default_image_size'):
        s = net.default_image_size
    else:
        s = 0
    if s not in size:
        size[s] = []
    size[s].append(n)

for k in sorted(size.keys()):
    print(k)
    pprint(sorted(size[k]))
