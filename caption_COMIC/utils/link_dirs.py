# -*- coding: utf-8 -*-
"""
Created on 27 Mar 2020 15:00:12

@author: jiahuei
"""
import os
import sys

pjoin = os.path.join
up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(up_dir(CURR_DIR))
sys.path.insert(1, BASE_DIR)
