#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:12:19 2017

@author: jiahuei
"""
import tensorflow as tf
# import numpy as np
import pickle
import os
import re
import time
import json
import logging
import common.ops_v1 as ops
from tqdm import tqdm
from json import encoder
from src.models import CaptionModel
from common.inputs import image_caption as cap
from common.mask_prune import pruning
from common.coco_caption.eval import evaluate_captions

logger = logging.getLogger(__name__)
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
pjoin = os.path.join

P_COCO = re.compile(r'(?<=_)\d+')
P_CKPT = re.compile(r'\d+')


def id_to_caption(ids, config):
    c = config
    captions = []
    if c.token_type == cap.TOKEN_RADIX:
        # Convert Radix IDs to sentence.
        base = c.radix_base
        word_len = c.radix_max_word_len
        for i in range(ids.shape[0]):
            row = [wid for wid in ids[i, :] if 0 <= wid < base]
            row = ops.grouper(row, word_len)
            sent = [c.radix_itow.get('_'.join(map(str, wid)), 'UNK') for wid in row]
            captions.append(' '.join(sent))
    else:
        # Convert word / char IDs to sentence.
        for i in range(ids.shape[0]):
            row = [wid for wid in ids[i, :] if wid >= 0 and wid != c.wtoi['<EOS>']]
            sent = [c.itow[str(wid)] for wid in row if str(wid) in c.itow]
            if c.token_type == 'word':
                captions.append(' '.join(sent))
            elif c.token_type == 'char':
                captions.append(''.join(sent))
    return captions


def run_inference(config, curr_ckpt_path):
    """
    Main inference function. Builds and executes the model.
    """
    c = config
    ckpt_dir, ckpt_file = os.path.split(curr_ckpt_path)
    ckpt_num = P_CKPT.findall(ckpt_file)[0]  # Checkpoint number
    
    # Setup input pipeline & Build model
    logger.debug('TensorFlow version: r{}'.format(tf.__version__))
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(c.rand_seed)
        inputs_man = cap.CaptionInput(c, is_inference=True)
        batch_size = c.batch_size_infer
        
        with tf.name_scope('infer'):
            m_infer = CaptionModel(
                c,
                mode='infer',
                batch_ops=inputs_man.batch_infer,
                reuse=False,
                name='inference')
        if c.supermask_type:
            with tf.name_scope('sparsity'):
                sampled_masks, masks = pruning.get_masks()
                total_sparsity, total_nnz, mask_sps = pruning.calculate_mask_sparsities(
                    sampled_masks, [m.op.name for m in masks])
                sparsity_ops = [total_sparsity, total_nnz, [s[1] for s in mask_sps]]
        init_fn = tf.local_variables_initializer()
        saver = tf.train.Saver()
    
    filenames = inputs_man.filenames_infer
    r = config.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    num_batches = int(c.split_sizes['infer'] / batch_size)
    
    raw_outputs = dict(captions={},
                       attention={},
                       filepath={},
                       beam_size=c.infer_beam_size,
                       max_caption_length=c.infer_max_length,
                       checkpoint_path=curr_ckpt_path,
                       checkpoint_number=ckpt_num)
    coco_json = []
    with sess:
        sess.run(init_fn)
        # Restore model from checkpoint
        logger.debug('Restoring from `{}`'.format(curr_ckpt_path))
        saver.restore(sess, curr_ckpt_path)
        g.finalize()
        
        if c.supermask_type:
            total_sparsity, total_nnz, sps = sess.run(sparsity_ops)
            sps_val = dict(global_step=ckpt_num,
                           total_sparsity=total_sparsity,
                           total_nnz=total_nnz,
                           mask_sps=list(zip([s[0] for s in mask_sps], sps)))
            pruning.write_sparsities_to_file(log_dir=c.infer_save_path, val=sps_val)
        
        logger.info('Graph constructed. Starting inference.')
        start_time = time.time()
        
        desc = 'Inference: checkpoint {}'.format(ckpt_num)
        for step in tqdm(range(num_batches), desc=desc, ncols=100):
            word_ids, attn_maps = sess.run(m_infer.infer_output)
            captions = id_to_caption(word_ids, c)
            # attn_maps = np.split(attn_maps, batch_size)
            
            # Get image ids, compile results
            batch_start = step * batch_size
            batch_end = (step + 1) * batch_size
            batch_filenames = filenames[batch_start: batch_end]
            
            for i, f in enumerate(batch_filenames):
                image_id = f.replace('.jpg', '')
                if '@' in image_id:
                    image_id = os.path.basename(image_id)
                else:
                    image_id = P_COCO.findall(image_id)
                    if isinstance(image_id, list) and len(image_id) > 0:
                        image_id = int(image_id[0])
                    else:
                        raise ValueError('Expected `image_id` to be list or string, saw `{}`'.format(type(image_id)))
                raw_outputs['captions'][image_id] = captions[i]
                # if c.infer_beam_size == 1:
                raw_outputs['attention'][image_id] = attn_maps[i]
                raw_outputs['filepath'][image_id] = f
                coco_json.append(dict(image_id=image_id, caption=str(captions[i])))
        
        logger.info("\nExample captions:\n{}\n".format("\n".join(captions[:3])))
        t = time.time() - start_time
        sess.close()
    
    # Ensure correctness
    assert len(filenames) == len(list(set(filenames)))
    assert len(filenames) == len(coco_json)
    assert len(filenames) == len(raw_outputs['attention'].keys())
    
    # Dump output files
    raw_output_fname = 'outputs___{}.pkl'.format(ckpt_num)
    coco_json_fname = 'captions___{}.json'.format(ckpt_num)
    
    # Captions with attention maps
    if c.save_attention_maps:
        with open(pjoin(c.infer_save_path, raw_output_fname), 'wb') as f:
            pickle.dump(raw_outputs, f, pickle.HIGHEST_PROTOCOL)
    # Captions with image ids
    with open(pjoin(c.infer_save_path, coco_json_fname), 'w') as f:
        json.dump(coco_json, f)
    if not os.path.isfile(pjoin(c.infer_save_path, 'infer_speed.txt')):
        out = ['Using GPU #: {}'.format(c.gpu),
               'Inference batch size: {}'.format(c.batch_size_infer),
               'Inference beam size: {}'.format(c.infer_beam_size),
               '']
        with open(pjoin(c.infer_save_path, 'infer_speed.txt'), 'a') as f:
            f.write('\r\n'.join(out))
    with open(pjoin(c.infer_save_path, 'infer_speed.txt'), 'a') as f:
        f.write('\r\n{}'.format(len(filenames) / t))
    logger.info("INFO: Inference completed. Time taken: {:4.2f} mins\n".format(t / 60))


def evaluate_model(config,
                   curr_ckpt_path,
                   scores_combined,
                   valid_ppl_dict=None,
                   test_ppl_dict=None):
    """
    Evaluates the model and returns the metric scores.
    """
    c = config
    
    ckpt_dir, ckpt_file = os.path.split(curr_ckpt_path)
    ckpt_num = int(P_CKPT.findall(ckpt_file)[0])
    output_filename = 'captions___{}.json'.format(ckpt_num)
    coco_json = pjoin(c.infer_save_path, output_filename)
    
    if c.run_inference:
        if not os.path.isfile('{}.index'.format(curr_ckpt_path)):
            logger.warning('`{}.index` not found. Checkpoint skipped.'.format(ckpt_file))
            return None
        if os.path.isfile(coco_json):
            logger.info('Found caption file `{}`. Skipping inference.'.format(
                os.path.basename(coco_json)))
        else:
            # Beam search to obtain captions
            run_inference(config, curr_ckpt_path)
    
    if not c.get_metric_score:
        return None
    
    # Evaluate captions
    print('')
    logger.info('Evaluation: checkpoint \t {}\n'.format(ckpt_num))
    
    results = evaluate_captions(coco_json, c.annotations_file)
    
    # Compile scores
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    scores = ['{:1.3f}'.format(results[m]) for m in metrics]
    scores_str = ['{}: {:1.3f}'.format(m, results[m]) for m in metrics]
    scores_combined[ckpt_num] = results
    
    valid_ckpt_missing = valid_ppl_dict is None or ckpt_num not in valid_ppl_dict
    test_ckpt_missing = test_ppl_dict is None or ckpt_num not in test_ppl_dict
    score_file = pjoin(c.infer_save_path, 'metric_scores')
    
    # Finally write aggregated scores to file
    with open(score_file + ".txt", 'a') as f:
        out_string = "===================================\r\n"
        out_string += "%s\r\n" % ckpt_file
        out_string += "Beam size: %d\r\n" % c.infer_beam_size
        out_string += "===================================\r\n"
        out_string += "%s\r\n" % "\r\n".join(scores_str)
        out_string += "Perplexity (valid): "
        
        if valid_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % valid_ppl_dict[ckpt_num]
        
        out_string += "Perplexity (test): "
        if test_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[ckpt_num]
        out_string += "\r\n\r\n"
        f.write(out_string)
    
    # Write scores to file in CSV style
    with open(score_file + ".csv", 'a') as f:
        out_string = "%d," % ckpt_num
        out_string += "%s," % ",".join(scores)
        if valid_ckpt_missing:
            out_string += "N/A,"
        else:
            out_string += "%2.3f," % valid_ppl_dict[ckpt_num]
        if test_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[ckpt_num]
        f.write(out_string)
    
    # Write individual scores
    # _dprint("results['evalImgs'] is: {}".format(results['evalImgs']))
    sorted_cider = sorted(results['evalImgs'],
                          key=lambda k: k['CIDEr'],
                          reverse=True)
    json_file = score_file + "_detailed_{}.json".format(ckpt_num)
    with open(json_file, 'w') as f:
        json.dump(sorted_cider, f)
    
    return scores_combined
