# -*- coding: utf-8 -*-
"""
Created on 31 Mar 2020 23:02:57

@author: jiahuei
"""
import os
from common.coco_caption.pycocotools.coco import COCO
from common.coco_caption.pycocoevalcap.eval import COCOEvalCap

pjoin = os.path.join
up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(up_dir(CURR_DIR))


def evaluate_captions(res_file, ann_file):
    default_ann_dir = pjoin(BASE_DIR, 'common', 'coco_caption', 'annotations')
    default_res_dir = pjoin(BASE_DIR, 'common', 'coco_caption', 'results')
    # create coco object and cocoRes object
    coco = COCO(pjoin(default_ann_dir, ann_file))
    coco_res = coco.loadRes(pjoin(default_res_dir, res_file))
    
    # create cocoEval object by taking coco and coco_res
    coco_eval = COCOEvalCap(coco, coco_res)
    
    # evaluate on a subset of images
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_res.getImgIds()
    
    # evaluate results
    coco_eval.evaluate()
    
    results = {}
    for metric, score in coco_eval.eval.items():
        # print '%s: %.3f' % (metric, score)
        results[metric] = score
    results['evalImgs'] = coco_eval.evalImgs
    return results
