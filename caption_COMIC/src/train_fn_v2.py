#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:44:23 2017

@author: jiahuei
"""
import tensorflow as tf
import numpy as np
import os
import sys
import time
import logging
import json
import traceback as tb
from pprint import pprint
from tqdm import tqdm
from tensorflow.python.client import timeline
from src import models
from src.infer_fn_v2 import id_to_caption
from common.scst.scorers_v1 import CaptionScorer
from common.mask_prune import pruning, masked_layer
from common.inputs.image_caption import CaptionInput, CaptionInputSCST
from common.inputs.mnist import MNISTInput
from common.inputs.vqa_v2 import VQAInput
# from common.vqa.eval import evaluate_answers
from common import ops_v1 as ops

logger = logging.getLogger(__name__)
pjoin = os.path.join
value_summary = ops.add_value_summary


# noinspection PyTypeChecker
def add_args(parser):
    # Main
    parser.add_argument(
        '--legacy', type=bool, default=False,
        help='If True, will match settings as described in paper.')
    parser.add_argument(
        '--token_type', type=str, default='word',  # default='radix',
        choices=['radix', 'word', 'char'],
        help='The language model.')
    parser.add_argument(
        '--radix_base', type=int, default=256,
        help='The base for Radix models.')

    # CNN
    parser.add_argument(
        '--cnn_name', type=str, default='inception_v1',
        help='The CNN model name.')
    parser.add_argument(
        '--cnn_input_size', type=ops.convert_int_csv, default='224,224',
        help='The network input size.')
    parser.add_argument(
        '--cnn_input_augment', type=bool, default=True,
        help='Whether to augment input images.')
    parser.add_argument(
        '--cnn_fm_attention', type=str, default='Mixed_4f',
        help='String, name of feature map for attention.')
    parser.add_argument(
        '--cnn_fm_projection', type=ops.convert_none, default='independent',  # default='tied',
        choices=[None, 'independent', 'tied'],
        help='String, feature map projection, from `none`, `independent`, `tied`.')

    # RNN
    parser.add_argument(
        '--rnn_name', type=str, default='LSTM',
        choices=['LSTM', 'LN_LSTM', 'GRU'],
        help='The type of RNN, from `LSTM`, `LN_LSTM` and `GRU`.')
    parser.add_argument(
        '--rnn_size', type=int, default=512,
        help='Int, number of RNN units.')
    parser.add_argument(
        '--rnn_word_size', type=int, default=256,
        help='The word size.')
    parser.add_argument(
        '--rnn_init_method', type=str, default='project_hidden',
        choices=['project_hidden', 'first_input'],
        help='The RNN init method.')
    parser.add_argument(
        '--rnn_keep_prob', type=float, default=0.65,
        help='Float, The keep rate for RNN input and output dropout.')
    parser.add_argument(
        '--rnn_recurr_dropout', type=bool, default=False,
        help='Whether to enable variational recurrent dropout.')

    # Attention
    parser.add_argument(
        '--attn_num_heads', type=int, default=1,  # default=8,
        help='The number of attention heads.')
    parser.add_argument(
        '--attn_size', type=int, default=None,
        help='The size of context vector.')
    parser.add_argument(
        '--attn_context_layer', type=bool, default=False,
        help='If True, add linear projection after multi-head attention.')
    parser.add_argument(
        '--attn_alignment_method', type=str, default='add',
        choices=['add_LN', 'add', 'dot'],
        help='Str, The alignment method / composition method.')
    parser.add_argument(
        '--attn_probability_fn', type=str, default='softmax',
        choices=['softmax', 'sigmoid'],
        help='Str, The attention map probability function.')
    parser.add_argument(
        '--attn_keep_prob', type=float, default=0.9,
        help='Float, The keep rate for attention map dropout.')
    parser.add_argument(
        '--attn_map_loss_scale', type=float, default=0.,
        help='Float, Loss weightage for attention map loss.')

    # Supermask
    parser.add_argument(
        '--supermask_type', type=ops.convert_none, default='none',
        choices=[None] + masked_layer.VALID_MASKS,
        help='The type of Supermask.')
    parser.add_argument(
        '--supermask_init_value', type=float, default=5.0,
        help='Init value for Supermasks.')
    parser.add_argument(
        '--supermask_sparsity_target', type=float, default=0.8,
        help='Desired sparsity.')
    parser.add_argument(
        '--supermask_sparsity_loss_fn', type=str, default='L1',
        choices=pruning.LOSS_TYPE,
        help='Loss function used to control the sparsity.')
    parser.add_argument(
        '--supermask_sparsity_weight', type=float, default=-1.0,
        help='Loss weightage used to control the sparsity.')
    parser.add_argument(
        '--supermask_loss_anneal', type=bool, default=True,
        help='If True, anneal Supermask loss using inverted cosine curve.')
    parser.add_argument(
        '--supermask_lr_start', type=float, default=1e2,
        help='Starting learning rate for Supermasks.')
    # parser.add_argument(
    #     '--supermask_train_strategy', type=str, default='none',
    #     choices=[None, 'sep_constant', 'sep'],
    #     help='Training strategy for masks: `none`, `sep_constant`, `sep`.')
    parser.add_argument(
        '--prune_freeze_scopes', type=ops.convert_csv_str_list, default='',
        help='Scopes to freeze pruning masks. Comma-separated.')

    # Training
    parser.add_argument(
        '--resume_training', type=bool, default=False,
        help='If True, resume training from latest checkpoint.')
    parser.add_argument(
        '--initialiser', type=str, default='xavier_uniform',
        choices=['xavier_uniform', 'xavier_normal', 'he', 'truncated_normal'],
        help='Initialiser: `xavier_uniform`, `xavier_normal`, `he`.')
    parser.add_argument(
        '--optimiser', type=str, default='adam',
        choices=['adam', 'sgd'],
        help='Optimiser: `adam`, `sgd`.')
    parser.add_argument(
        '--batch_size_train', type=int, default=32,
        help='Batch size for training.')
    # Divisors of 25010: 1, 2, 5, 10, 41, 61, 82, 122, 205, 305, 410, 610, 2501, 5002, 12505, 25010
    parser.add_argument(
        '--batch_size_eval', type=int, default=-1,
        help='Batch size for validation.')
    parser.add_argument(
        '--max_epoch', type=int, default=30,
        help='The max epoch training.')
    parser.add_argument(
        '--lr_start', type=float, default=1e-2,
        help='Float, determines the starting learning rate.')
    parser.add_argument(
        '--lr_end', type=float, default=1e-5,
        help='Float, determines the ending learning rate.')
    parser.add_argument(
        '--clip_gradient_norm', type=float, default=0.,
        help='Float, if greater than 0 then the gradients would be clipped by it (L2 norm).')
    parser.add_argument(
        '--cnn_grad_multiplier', type=float, default=1.0,
        help='Float, determines the gradient multiplier when back-prop thru CNN.')
    parser.add_argument(
        '--adam_epsilon', type=float, default=1e-2,
        help='Float, determines the epsilon value of ADAM.')
    parser.add_argument(
        '--l2_decay', type=float, default=1e-5,
        help='Float, determines the L2 decay weight.')
    parser.add_argument(
        '--freeze_scopes', type=ops.convert_csv_str_list, default='Model/encoder/cnn',
        help='The scopes to freeze / do not train.')
    parser.add_argument(
        '--add_image_summaries', type=bool, default=False,
        help='If True, add image summaries.')
    parser.add_argument(
        '--add_vars_summaries', type=bool, default=False,
        help='If True, add trainable variable summaries.')
    parser.add_argument(
        '--add_grad_summaries', type=bool, default=False,
        help='If True, add gradient summaries.')

    # Misc
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='The gpu number.')
    parser.add_argument(
        '--verbosity', type=int, default=10, choices=[10, 20])
    parser.add_argument(
        '--capture_profile', type=bool, default=False,
        help='If True, run profiling and collect full trace.')

    return parser


def get_log_name(args):
    if args.token_type == 'radix':
        token = 'radix_b{}'.format(args.radix_base)
    else:
        token = args.token_type
    name = '_'.join([
        token,
        'w{}'.format(args.rnn_word_size),
        args.rnn_name,
        'r{}'.format(args.rnn_size),
        'h{}'.format(args.attn_num_heads),
        args.cnn_fm_projection[:3] if args.cnn_fm_projection else 'none',
    ])
    if args.legacy:
        name = 'legacy_' + name

    if args.supermask_type:
        if 'prune' in args.log_root:
            if '_' in args.initialiser:
                name += '_{}'.format(''.join(i[0] for i in args.initialiser.split('_')))
            else:
                name += '_{}'.format(args.initialiser[:2])

            if args.supermask_type in masked_layer.SUPER_MASKS:
                name += '_{}_{:2.1e}_init_{:2.1f}_{}_wg_{:2.1f}'.format(
                    args.supermask_type[:3].upper(),
                    args.supermask_lr_start,
                    args.supermask_init_value,
                    args.supermask_sparsity_loss_fn,
                    args.supermask_sparsity_weight)
                name += '_ann' if args.supermask_loss_anneal else ''
            else:
                name = name + '_' + args.supermask_type[0] + ''.join(x.title() for x in args.supermask_type.split('_'))[
                                                             1:]
        name += '_sps_{:4.3f}'.format(args.supermask_sparsity_target)
    return name


def get_defaults(args):
    rand_seed = [0,
                 48964896,
                 88888888,
                 123456789]
    try:
        rand_seed = rand_seed[args.run]
    except IndexError:
        rand_seed = args.run

    defaults = dict(
        rand_seed=rand_seed,
        # Main
        legacy=False,
        token_type='word',
        radix_base=256,
        # CNN
        cnn_name='inception_v1',
        cnn_input_size=[224, 224],
        cnn_input_augment=True,
        cnn_fm_attention='Mixed_4f',
        cnn_fm_projection='independent',
        # RNN
        rnn_name='LSTM',
        rnn_size=512,
        rnn_word_size=256,
        rnn_init_method='project_hidden',
        rnn_keep_prob=0.65,
        rnn_recurr_dropout=False,
        # Attention
        attn_num_heads=1,
        attn_size=None,
        attn_context_layer=False,
        attn_alignment_method='add',
        attn_probability_fn='softmax',
        attn_keep_prob=0.9,
        attn_map_loss_scale=0.0,
        # Supermask
        supermask_type=None,
        supermask_init_value=5.0,
        supermask_sparsity_target=0.8,
        supermask_sparsity_loss_fn='L1',
        supermask_sparsity_weight=-1.0,
        supermask_loss_anneal=True,
        supermask_lr_start=100.0,
        prune_freeze_scopes=None,
        # Training
        resume_training=False,
        initialiser='xavier_uniform',
        optimiser='adam',
        batch_size_train=32,
        batch_size_eval=-1,
        max_epoch=30,
        lr_start=0.01,
        lr_end=1e-05,
        cnn_grad_multiplier=1.0,
        adam_epsilon=0.01,
        l2_decay=1e-05,
        freeze_scopes=['Model/encoder/cnn'],
        add_image_summaries=False,
        add_vars_summaries=False,
        add_grad_summaries=False,
        # Misc
        gpu=0,
        verbosity=10,
        capture_profile=False,
    )
    return defaults


def train_caption_xe(config):
    """Main training function. To be called by `try_to_train()`."""

    c = config
    logger.debug('TensorFlow version: r{}'.format(tf.__version__))
    logger.info('Logging to `{}`.'.format(c.log_path))

    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(c.rand_seed)
        inputs = CaptionInput(c)
        c.save_config_to_file()

        lr = c.lr_start
        num_batches = int(c.split_sizes['train'] / c.batch_size_train)
        n_steps_log = int(num_batches / c.num_logs_per_epoch)

        with tf.name_scope('train'):
            m_train = models.CaptionModel(config=c,
                                          mode='train',
                                          batch_ops=inputs.batch_train,
                                          reuse=False,
                                          name='train')

        with tf.name_scope('valid'):
            m_valid = models.CaptionModel(config=c,
                                          mode='eval',
                                          batch_ops=inputs.batch_eval,
                                          reuse=True,
                                          name='valid')

        # Apply masks
        if c.supermask_type == masked_layer.LOTTERY:
            _m = m_train.all_pruning_masks
            _w = m_train.all_pruned_weights
            assert [_.op.name.replace("/mask", "") for _ in _m] == [_.op.name for _ in _w]
            assert len(_m) > 0
            with tf.name_scope('apply_masks'):
                _mo = [tf.multiply(m, w) for m, w in zip(_m, _w)]
                weight_assign_ops = [tf.assign(w, w_m) for w, w_m in zip(_w, _mo)]

        init_fn = tf.global_variables_initializer()
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
        model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
        saver = tf.train.Saver(max_to_keep=2)
        if c.capture_profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            trace_logdir = pjoin(c.log_path, 'trace')
            if not os.path.exists(trace_logdir):
                os.makedirs(trace_logdir)

    r = c.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    summary_writer = tf.summary.FileWriter(c.log_path, g)

    with sess:
        # Restore model from checkpoint if provided
        sess.run(init_fn)
        lr = m_train.restore_model(sess, saver, lr)
        # Maybe prune
        m0, _, _ = m_train.maybe_run_assign_masks(sess)
        # Lottery ticket resets weights to init
        if c.supermask_type == masked_layer.LOTTERY:
            lt_var_list = tf.contrib.framework.filter_variables(
                var_list=model_vars,
                include_patterns=["Model"],
                exclude_patterns=["mask"],
                reg_search=True
            )
            assert len(lt_var_list) > 0
            lt_saver = tf.train.Saver(var_list=lt_var_list)
            lt_saver.restore(sess, os.path.join(c.checkpoint_path, 'model_compact-0'))
        # Save pruned network prior to retraining
        # We use a different Saver to avoid checkpoint auto-deletion
        pruned_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
        pruned_saver.save(sess, c.save_path + '_compact', 0)

        g.finalize()
        ops.get_model_size(scope_or_list='Model/decoder', log_path=c.log_path)
        ops.get_model_size(scope_or_list=m_train.final_trainable_variables, log_path=c.log_path)
        start_step = sess.run(m_train.global_step)

        training_ops = [m_train.model_loss,
                        m_train.global_step,
                        m_train.lr,
                        m_train.summary_op]
        run_kwargs = dict(options=run_options, run_metadata=run_metadata) if c.capture_profile else {}

        logger.info('Training begins now.')
        start_epoch = t_log = time.time()

        for step in range(start_step, c.max_step):
            epoch = int(step / num_batches) + 1
            # Write summary to disk once every `n_steps_log` steps
            if (step + 1) % n_steps_log == 0:
                ppl, global_step, lr, summary = sess.run(training_ops)
                speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
                t_log = time.time()
                print('   Training speed: {:7.2f} examples/sec.'.format(speed))
                summary_writer.add_summary(summary, global_step)
                value_summary({'train/speed': speed}, summary_writer, global_step)

            # Quick logging
            elif (step + 1) % int(n_steps_log / 5) == 0:
                ppl, global_step, lr = sess.run(training_ops[:3])
                ppl = np.exp(ppl)
                logstr = 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
                    epoch, ((step % num_batches) + 1) / num_batches * 100)
                logstr += 'Perplexity {:8.4f} ~ LR {:5.3e} ~ '.format(ppl, lr)
                logstr += 'Step {}'.format(global_step)
                print('   ' + logstr)
            else:
                if c.capture_profile:
                    if step <= 20:
                        # summary_writer.add_run_metadata(run_metadata, 'step_{}'.format(global_step), global_step)
                        # This saves the timeline to a chrome trace format:
                        ppl, global_step = sess.run(training_ops[:2], **run_kwargs)
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(pjoin(trace_logdir, 'timeline_{}.json').format(global_step), 'w') as f:
                            f.write(chrome_trace)
                    else:
                        sys.exit('Profiling complete.')
                else:
                    ppl, global_step = sess.run(training_ops[:2])

            if num_batches > 5000:
                save = (step + 1) % int(num_batches / 2) == 0
            else:
                save = (step + 1) % num_batches == 0
            save = save and (step + 100) < c.max_step

            # Evaluation and save model
            if save or (step + 1) == c.max_step:
                model_saver.save(sess, c.save_path + '_compact', global_step)
                saver.save(sess, c.save_path, global_step)
                _eval_perplexity(sess, c, m_valid, summary_writer, global_step)

                # _mask_check = c.supermask_type in masked_layer.MAG_ANNEAL and c.train_mode == 'cnn_freeze'
                if c.supermask_type in masked_layer.MAG_HARD:
                    # Debug: ensure mask is not updated
                    # TODO: remove when confident with correctness
                    m1 = sess.run(m_train.masks)
                    for i, x in enumerate(m1):
                        assert np.all(m0[i] == x)

            if (step + 1) % num_batches == 0:
                if c.legacy:
                    lr = _lr_reduce_check(config, epoch, lr)
                    m_train.update_lr(sess, lr)
                    sess.run(m_train.lr)
                t = time.time() - start_epoch
                print('\n\n>>> Epoch {:3d} complete'.format(epoch))
                print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
                start_epoch = time.time()
            time.sleep(0.055)  # throttle for thermal protection

        # Apply masks in order to support iterative pruning of Lottery tickets
        if c.supermask_type == masked_layer.LOTTERY:
            sess.run(weight_assign_ops)
        sess.close()
        print('\n')
        logger.info('Training completed.')


def train_caption_scst(config, idx_ngram=False):
    """SCST training function. To be called by `try_to_train()`."""

    logger.debug('TensorFlow version: r{}'.format(tf.__version__))
    logger.info('Logging to `{}`.'.format(config.log_path))

    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)
        inputs = CaptionInputSCST(config)
        c = inputs.config
        c.save_config_to_file()

        lr = c.lr_start
        num_batches = int(c.split_sizes['train'] / c.batch_size_train)
        n_steps_log = int(num_batches / c.num_logs_per_epoch)

        with tf.name_scope('train'):
            m_train = models.CaptionModelSCST(config=c,
                                              scst_mode='train',
                                              reuse=False)

        with tf.name_scope('sample'):
            m_sample = models.CaptionModelSCST(config=c,
                                               scst_mode='sample',
                                               reuse=True)

        init_fn = tf.global_variables_initializer()
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
        model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
        saver = tf.train.Saver(max_to_keep=2)

    r = c.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    summary_writer = tf.summary.FileWriter(c.log_path, g)

    if idx_ngram:
        idf_fname = 'scst-idxs'
    else:
        idf_fname = 'scst-words'
    idf_fname = c.dataset_file_pattern.format(idf_fname) + '.p'
    idf_fp = pjoin(c.dataset_dir, 'captions', idf_fname)
    if not os.path.isfile(idf_fp):
        raise ValueError('File not found: `{}`'.format(idf_fp))
    wg = dict(ciderD=c.scst_weight_ciderD, bleu=c.scst_weight_bleu)
    scorer = CaptionScorer(path_to_cached_tokens=idf_fp, metric_weights=wg)

    with sess:
        # Restore model from checkpoint if provided
        sess.run(init_fn)
        lr = m_train.restore_model(sess, saver, lr)
        g.finalize()
        ops.get_model_size(scope_or_list='Model/decoder', log_path=c.log_path)
        ops.get_model_size(scope_or_list=m_train.final_trainable_variables, log_path=c.log_path)
        start_step = sess.run(m_train.global_step)
        greedy_high_sc = 0

        sampling_ops = [m_sample.dec_preds_beam, m_sample.dec_preds_greedy]  # ,
        #               m_sample.dec_preds_sample]
        training_ops = [m_train.train_scst,
                        m_train.global_step,
                        m_train.lr,
                        m_train.summary_op, ]

        logger.info('Graph constructed. SCST training begins now.')
        start_epoch = t_log = time.time()

        for step in range(start_step, c.max_step):
            epoch = int(step / num_batches) + 1

            # Retrieve model inputs
            imgs, refs = sess.run(inputs.batch_train)

            # Generate captions
            # `cap_beam` is of shape (beam_size, batch_size, time)
            # `cap_beam` is then reshaped to (beam_size * batch_size, time)
            # after converting to string, the structure of `cap_beam` will be
            # [[im0_hypo0], ..., [imN_hypo0], [im0_hypo1], ..., [imN_hypo1], ...]
            # cap_beam, cap_greedy, cap_sample = sess.run(sampling_ops, feed_dict={m_sample.imgs: imgs})
            cap_beam, cap_greedy = sess.run(sampling_ops, feed_dict={m_sample.imgs: imgs})
            cap_beam = np.reshape(cap_beam, [-1, cap_beam.shape[-1]])
            cap_beam = [[s] for s in id_to_caption(cap_beam, c)]
            cap_greedy = [[s] for s in id_to_caption(cap_greedy, c)]
            # cap_sample = [[s] for s in id_to_caption(cap_sample, c)]
            # cap_beam += cap_sample

            # Get RL rewards, convert string to padded numpy array
            hypos, sc_sample, sc_greedy = scorer.get_hypo_scores(refs.tolist(), cap_beam, cap_greedy)
            rewards = sc_sample - sc_greedy
            greedy_high_sc = max(greedy_high_sc, np.amax(sc_greedy))
            hypos_idx = inputs.captions_to_batched_ids(hypos)
            assert hypos_idx.shape[0] == sc_sample.shape[0]

            # Train the model
            imgs = np.concatenate([imgs] * max(c.scst_beam_size, 1))
            feed = {m_train.imgs: imgs,
                    m_train.captions: hypos_idx,
                    m_train.rewards: rewards}
            ppl, global_step, lr, summary = sess.run(training_ops, feed_dict=feed)

            # Write summary to disk once every `n_steps_log` steps
            if (step + 1) % n_steps_log == 0:
                speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
                t_log = time.time()
                summary_writer.add_summary(summary, global_step)
                value_summary({'train/speed': speed,
                               'score_wg/greedy': np.mean(sc_greedy),
                               'score_wg/beam': np.mean(sc_sample)},
                              summary_writer, global_step)
                logstr = ''
                logstr += '\n   Training speed: {:7.2f} examples/sec.'.format(speed)
                logstr += '\n   mean reward: \t{:8.4f}'.format(np.mean(rewards))
                logstr += '\n   greedy high score: \t{:8.4f}'.format(greedy_high_sc)
                logstr += '\n   greedy: \t\t`{}`'.format(cap_greedy[0][0])
                logstr += '\n   top beam: \t\t`{}`'.format(hypos[0][0])
                logstr += '\n'
                print(logstr)

            # Quick logging
            elif (step + 1) % int(n_steps_log / 5) == 0:
                # ppl = np.exp(ppl)
                logstr = '   '
                logstr += 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
                    epoch, ((step % num_batches) + 1) / num_batches * 100)
                logstr += 'Greedy score {:8.4f} ~ '.format(np.mean(sc_greedy))
                logstr += 'Loss {:8.4f} ~ '.format(ppl)
                logstr += 'LR {:5.3e} ~ '.format(lr)
                logstr += 'Step {}'.format(global_step)
                print(logstr)

            if num_batches > 5000:
                save = (step + 1) % int(num_batches / 2) == 0
            else:
                save = (step + 1) % num_batches == 0
            save = save and (step + 100) < c.max_step

            # Save model
            if save or (step + 1) == c.max_step:
                model_saver.save(sess, c.save_path + '_compact', global_step)
                saver.save(sess, c.save_path, global_step)

            if (step + 1) % num_batches == 0:
                t = time.time() - start_epoch
                print('\n\n>>> Epoch {:3d} complete'.format(epoch))
                print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
                start_epoch = time.time()

        sess.close()
        print('\n')
        logger.info('Training completed.')


def train_rnn_mnist(config):
    """Main training function. To be called by `try_to_train()`."""

    logger.debug('TensorFlow version: r{}'.format(tf.__version__))
    logger.info('Logging to `{}`.'.format(config.log_path))

    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)
        inputs = MNISTInput(config)
        c = inputs.config
        c.save_config_to_file()

        lr = c.lr_start
        num_batches = int(c.split_sizes['train'] / c.batch_size_train)
        n_steps_log = int(num_batches / c.num_logs_per_epoch)

        with tf.name_scope('train'):
            m_train = models.MNISTRNNModel(config=c,
                                           mode='train',
                                           batch_ops=inputs.batch_train,
                                           reuse=False,
                                           name='train')

        with tf.name_scope('valid'):
            m_valid = models.MNISTRNNModel(config=c,
                                           mode='eval',
                                           batch_ops=inputs.batch_valid,
                                           reuse=True,
                                           name='valid')

        with tf.name_scope('test'):
            m_test = models.MNISTRNNModel(config=c,
                                          mode='eval',
                                          batch_ops=inputs.batch_test,
                                          reuse=True,
                                          name='test')

        # Sparsity calculation ops
        with tf.name_scope('sparsity'):
            # weights = tf.contrib.model_pruning.get_weights()
            weights = pruning.get_weights()
            sampled_masks, masks = pruning.get_masks()
            if len(sampled_masks) > 0:
                masks_sparsity, _, _ = pruning.calculate_mask_sparsities(
                    sampled_masks, [m.op.name for m in masks])
                weights_sparsity, _, _ = pruning.calculate_weight_sparsities(
                    weights, [w.op.name for w in weights])
            else:
                masks_sparsity = weights_sparsity = None

        # Apply masks after training is completed for final verification
        with tf.name_scope('apply_masks'):
            masked_wg = [tf.multiply(m, w) for m, w in zip(sampled_masks, weights)]
            weight_assign_ops = [tf.assign(w, w_m) for w, w_m in zip(weights, masked_wg)]

        init_fn = tf.global_variables_initializer()
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
        model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
        saver = tf.train.Saver(max_to_keep=2)

    r = c.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    summary_writer = tf.summary.FileWriter(c.log_path, g)

    with sess:
        # Restore model from checkpoint if provided
        sess.run(init_fn)
        lr = m_train.restore_model(sess, saver, lr)
        # Maybe prune
        m0, _, _ = m_train.maybe_run_assign_masks(sess)
        # Save pruned network prior to retraining
        # We use a different Saver to avoid checkpoint auto-deletion
        # pruned_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
        # pruned_saver.save(sess, c.save_path + '_compact', 0)

        g.finalize()
        ops.get_model_size(scope_or_list=m_train.final_trainable_variables, log_path=c.log_path)
        start_step = sess.run(m_train.global_step)

        training_ops = [m_train.model_loss,
                        m_train.global_step,
                        m_train.lr,
                        m_train.summary_op]

        logger.info('Training begins now.')
        start_epoch = t_log = time.time()

        for step in range(start_step, c.max_step):
            epoch = int(step / num_batches) + 1

            # Write summary to disk once every `n_steps_log` steps
            if (step + 1) % n_steps_log == 0:
                ppl, global_step, lr, summary = sess.run(training_ops)
                speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
                t_log = time.time()
                print('   Training speed: {:7.2f} examples/sec.'.format(speed))
                summary_writer.add_summary(summary, global_step)
                value_summary({'train/speed': speed}, summary_writer, global_step)
            # Quick logging
            elif (step + 1) % int(n_steps_log / 5) == 0:
                ppl, global_step, lr = sess.run(training_ops[:3])
                ppl = np.exp(ppl)
                logstr = 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
                    epoch, ((step % num_batches) + 1) / num_batches * 100)
                logstr += 'Loss {:8.4f} ~ LR {:5.3e} ~ '.format(ppl, lr)
                logstr += 'Step {}'.format(global_step)
                print('   ' + logstr)
            else:
                ppl, global_step = sess.run(training_ops[:2])

            if num_batches > 5000:
                save = (step + 1) % int(num_batches / 2) == 0
            else:
                save = (step + 1) % num_batches == 0
            save = save and (step + 100) < c.max_step

            # Evaluation and save model
            if save or (step + 1) == c.max_step:
                model_saver.save(sess, c.save_path + '_compact', global_step)
                saver.save(sess, c.save_path, global_step)
                _eval_classification(sess, c, m_valid, masks_sparsity, summary_writer, global_step)
                _eval_classification(sess, c, m_test, masks_sparsity, summary_writer, global_step)

                # _mask_check = c.supermask_type in masked_layer.MAG_ANNEAL and c.train_mode == 'cnn_freeze'
                if c.supermask_type in masked_layer.MAG_HARD:
                    # Debug: ensure mask is not updated
                    # TODO: remove when confident with correctness
                    m1 = sess.run(m_train.masks)
                    for i, x in enumerate(m1):
                        assert np.all(m0[i] == x)

            if (step + 1) % num_batches == 0:
                if c.legacy:
                    lr = _lr_reduce_check(config, epoch, lr)
                    m_train.update_lr(sess, lr)
                    sess.run(m_train.lr)
                t = time.time() - start_epoch
                print('\n\n>>> Epoch {:3d} complete'.format(epoch))
                print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
                start_epoch = time.time()

        # Verify
        print('\n')
        logger.info('Training completed.')
        if weights_sparsity is not None:
            sps_before = sess.run(weights_sparsity)
            logger.info('Sparsifying weights...')
            sess.run(weight_assign_ops)
            sps_after = sess.run(weights_sparsity)
            logger.info('Sparsity before/after: {:9.7f} / {:9.7f}'.format(sps_before, sps_after))
            _eval_classification(sess, c, m_valid, weights_sparsity, summary_writer, None)
            _eval_classification(sess, c, m_test, weights_sparsity, summary_writer, None)
        sess.close()
        print('\n')


# def train_vqa(config):
#     c = config
#     logger.debug('TensorFlow version: r{}'.format(tf.__version__))
#     logger.info('Logging to `{}`.'.format(c.log_path))
#
#     # Setup input pipeline & Build model
#     g = tf.Graph()
#     with g.as_default():
#         tf.set_random_seed(c.rand_seed)
#         inputs = VQAInput(c)
#         c.save_config_to_file()
#
#         lr = c.lr_start
#         num_batches = int(c.split_sizes['train'] / c.batch_size_train)
#         n_steps_log = int(num_batches / c.num_logs_per_epoch)
#
#         with tf.name_scope('train'):
#             m_train = models.VQAModelSimple(config=c,
#                                         mode='train',
#                                         batch_ops=inputs.batch_train,
#                                         reuse=False,
#                                         name='train')
#         if inputs.batch_eval is not None:
#             with tf.name_scope('valid'):
#                 m_valid = models.VQAModelSimple(config=c,
#                                             mode='eval',
#                                             batch_ops=inputs.batch_eval,
#                                             reuse=True,
#                                             name='valid')
#
#         init_fn = tf.global_variables_initializer()
#         model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
#         model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
#         saver = tf.train.Saver(max_to_keep=2)
#
#     r = c.per_process_gpu_memory_fraction
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
#     summary_writer = tf.summary.FileWriter(c.log_path, g)
#
#     with sess:
#         # Restore model from checkpoint if provided
#         sess.run(init_fn)
#         lr = m_train.restore_model(sess, saver, lr)
#         # Maybe prune
#         m0, _, _ = m_train.maybe_run_assign_masks(sess)
#         # Save pruned network prior to retraining
#         # We use a different Saver to avoid checkpoint auto-deletion
#         # pruned_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
#         # pruned_saver.save(sess, c.save_path + '_compact', 0)
#
#         g.finalize()
#         ops.get_model_size(scope_or_list=m_train.final_trainable_variables, log_path=c.log_path)
#         ops.get_model_size(scope_or_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), log_path=c.log_path)
#         start_step = sess.run(m_train.global_step)
#
#         training_ops = [m_train.model_loss,
#                         m_train.global_step,
#                         m_train.accuracy,
#                         m_train.lr,
#                         m_train.summary_op]
#
#         logger.info('Training begins now.')
#         start_epoch = t_log = time.time()
#
#         for step in range(start_step, c.max_step):
#             epoch = int(step / num_batches) + 1
#             # if (step + 1) % n_steps_log == 0:
#             #     # images, questions, answers, question_id = sess.run(inputs.batch_eval)
#             #     # from PIL import Image
#             #     # for z, _ in enumerate(images):
#             #     #     _ = (_ + 1.) * (255. / 2.)
#             #     #     print(_.shape)
#             #     #     _ = Image.fromarray(_.astype(np.uint8))
#             #     #     _.save(pjoin(c.log_path, str(z) + '.jpg'))
#             #     # print('images:')
#             #     # pprint(images)
#             #     # print('questions:')
#             #     # pprint(questions)
#             #     # print('answers:')
#             #     # pprint(answers)
#             #     # print('question_id')
#             #     # pprint(question_id)
#             #     encoder_inputs, answers = sess.run([m_train.encoder_inputs, m_train.answers])
#             #     print('encoder_inputs:')
#             #     pprint(encoder_inputs['input_seq'][:, 0, 0])
#             #     pprint(encoder_inputs['input_seq'].shape)
#             #     print('answers')
#             #     pprint(np.argmax(answers, axis=-1))
#             #     pprint(answers.shape)
#             #     speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
#             #     t_log = time.time()
#             #     print('   Speed: {:7.2f} examples/sec.'.format(speed))
#             #     print('Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
#             #         epoch, ((step % num_batches) + 1) / num_batches * 100))
#             #     # exit()
#             # else:
#             #     sess.run(inputs.batch_train)
#             # Write summary to disk once every `n_steps_log` steps
#             if (step + 1) % n_steps_log == 0:
#                 loss, global_step, acc, lr, summary, predictions = sess.run(training_ops + [m_train.predictions])
#                 speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
#                 t_log = time.time()
#                 print('   Training speed: {:7.2f} examples/sec.'.format(speed))
#                 summary_writer.add_summary(summary, global_step)
#                 value_summary({'train/speed': speed}, summary_writer, global_step)
#                 print(predictions[:15])
#                 # _eval_vqa(sess, c, m_valid, summary_writer, 0)
#
#             # Quick logging
#             elif (step + 1) % int(n_steps_log / 5) == 0:
#                 loss, global_step, acc, lr = sess.run(training_ops[:4])
#                 logstr = 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
#                     epoch, ((step % num_batches) + 1) / num_batches * 100)
#                 logstr += 'Loss {:8.4f} ~ Accuracy {:5.2f} ~ LR {:5.3e} ~ '.format(loss, acc, lr)
#                 logstr += 'Step {}'.format(global_step)
#                 print('   ' + logstr)
#             else:
#                 loss, global_step = sess.run(training_ops[:2])
#
#             # Evaluation and save model
#             save = (step + 1) % num_batches == 0 and (step + 100) < c.max_step
#             if save or (step + 1) == c.max_step:
#                 model_saver.save(sess, c.save_path + '_compact', global_step)
#                 saver.save(sess, c.save_path, global_step)
#                 _eval_vqa(sess, c, m_valid, summary_writer, global_step)
#
#                 # _mask_check = c.supermask_type in masked_layer.MAG_ANNEAL and c.train_mode == 'cnn_freeze'
#                 if c.supermask_type in masked_layer.MAG_HARD:
#                     # Debug: ensure mask is not updated
#                     # TODO: remove when confident with correctness
#                     m1 = sess.run(m_train.masks)
#                     for i, x in enumerate(m1):
#                         assert np.all(m0[i] == x)
#
#             if (step + 1) % num_batches == 0:
#                 t = time.time() - start_epoch
#                 print('\n\n>>> Epoch {:3d} complete'.format(epoch))
#                 print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
#                 start_epoch = time.time()
#
#         sess.close()
#         print('\n')
#         logger.info('Training completed.')


# def train_caption_ae(config):
#     """Main training function. To be called by `try_to_train()`."""
#
#     c = config
#     logger.debug('TensorFlow version: r{}'.format(tf.__version__))
#     logger.info('Logging to `{}`.'.format(c.log_path))
#
#     # Setup input pipeline & Build model
#     g = tf.Graph()
#     with g.as_default():
#         tf.set_random_seed(c.rand_seed)
#         # inputs = inputs.CaptionInputAE(c)
#         inputs = inputs.CaptionInputVAEX(c)
#         c.save_config_to_file()
#
#         num_batches = int(c.split_sizes['train'] / c.batch_size_train)
#         lr = c.lr_start
#         n_steps_log = int(num_batches / c.num_logs_per_epoch)
#
#         with tf.name_scope('train'):
#             m_train = CaptionModelVAEX(config=c,
#                                        mode='train',
#                                        batch_ops=inputs.batch_train,
#                                        reuse=False,
#                                        name='train')
#
#         with tf.name_scope('valid'):
#             m_valid = CaptionModelVAEX(config=c,
#                                        mode='eval',
#                                        batch_ops=inputs.batch_eval,
#                                        reuse=True,
#                                        name='valid')
#
#         init_fn = tf.global_variables_initializer()
#         model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
#         model_saver = tf.train.Saver(var_list=model_vars, max_to_keep=c.max_saves)
#         saver = tf.train.Saver(max_to_keep=2)
#
#     r = c.per_process_gpu_memory_fraction
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
#     summary_writer = tf.summary.FileWriter(c.log_path, g)
#
#     with sess:
#         # Restore model from checkpoint if provided
#         sess.run(init_fn)
#         lr = m_train.restore_model(sess, saver, lr)
#         g.finalize()
#         ops.get_model_size(scope_or_list=m_train.final_trainable_variables, log_path=c.log_path)
#         start_step = sess.run(m_train.global_step)
#
#         training_ops = [m_train.model_loss,
#                         m_train.global_step,
#                         m_train.lr,
#                         m_train.summary_op]
#         run_kwargs = {}
#
#         logger.info('Training begins now.')
#         start_epoch = t_log = time.time()
#
#         for step in range(start_step, c.max_step):
#             epoch = int(step / num_batches) + 1
#             # Write summary to disk once every `n_steps_log` steps
#             if (step + 1) % n_steps_log == 0:
#                 ppl, global_step, lr, summary = sess.run(training_ops)
#                 speed = n_steps_log * c.batch_size_train / (time.time() - t_log)
#                 t_log = time.time()
#                 print('   Training speed: {:7.2f} examples/sec.'.format(speed))
#                 summary_writer.add_summary(summary, global_step)
#                 value_summary({'train/speed': speed}, summary_writer, global_step)
#             # Quick logging
#             elif (step + 1) % int(n_steps_log / 5) == 0:
#                 ppl, global_step, lr = sess.run(training_ops[:3])
#                 ppl = np.exp(ppl)
#                 logstr = 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
#                     epoch, ((step % num_batches) + 1) / num_batches * 100)
#                 logstr += 'Perplexity {:8.4f} ~ LR {:5.3e} ~ '.format(ppl, lr)
#                 logstr += 'Step {}'.format(global_step)
#                 print('   ' + logstr)
#             else:
#                 ppl, global_step = sess.run(training_ops[:2])
#
#             if num_batches > 5000:
#                 save = (step + 1) % int(num_batches / 2) == 0
#             else:
#                 save = (step + 1) % num_batches == 0
#             save = save and (step + 100) < c.max_step
#
#             # Evaluation and save model
#             if save or (step + 1) == c.max_step:
#                 model_saver.save(sess, c.save_path + '_compact', global_step)
#                 saver.save(sess, c.save_path, global_step)
#                 _eval_perplexity(sess, c, m_valid, summary_writer, global_step)
#
#             if (step + 1) % num_batches == 0:
#                 if c.legacy:
#                     lr = _lr_reduce_check(config, epoch, lr)
#                     m_train.update_lr(sess, lr)
#                     sess.run(m_train.lr)
#                 t = time.time() - start_epoch
#                 print('\n\n>>> Epoch {:3d} complete'.format(epoch))
#                 print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
#                 start_epoch = time.time()
#
#         sess.close()
#         print('\n')
#         logger.info('Training completed.')


def _lr_reduce_check(config, epoch, learning_rate):
    """ Helper to reduce learning rate every n epochs."""
    if (learning_rate > config.lr_end
            and epoch % config.lr_reduce_every_n_epochs == 0):
        learning_rate /= 2
        if learning_rate < config.lr_end:
            learning_rate = config.lr_end
    return learning_rate


def _eval_perplexity(session, c, m, summary_writer, global_step):
    """
    Wrapper for running the validation loop.
    Returns the average perplexity per word.
    """
    name = m.name
    split_size = c.split_sizes[name]
    assert split_size % c.batch_size_eval == 0
    num_batches = int(split_size / c.batch_size_eval)
    ppl_list = []
    print('\nEvaluating model...\n')

    for step in tqdm(range(num_batches), desc='evaluation', ncols=100):
        ppl = session.run(m.model_loss)
        ppl_list.append(ppl)
    avg_ppl = np.exp(np.mean(ppl_list))
    print('>>> {} perplexity per word: {:8.4f}\n'.format(name, avg_ppl))
    value_summary({'{}/perplexity'.format(name): avg_ppl}, summary_writer, global_step)
    return avg_ppl


def _eval_classification(session, c, m, sps_op, summary_writer, global_step):
    """
    Wrapper for running the validation loop of MNIST.
    """
    name = m.name
    batch_size = c.batch_size_eval
    split_size = c.split_sizes[name]
    assert split_size % batch_size == 0
    num_batches = int(split_size / batch_size)
    print('\nEvaluating model...\n')

    preds = np.zeros(shape=[split_size])
    labels = np.zeros(shape=[split_size])
    if sps_op is None:
        sps = 0.
    else:
        sps = session.run(sps_op)
    for step in tqdm(range(num_batches), desc='evaluation', ncols=100):
        logits, lbl = session.run([m.dec_logits, m.batch_ops[1]])
        labels[step * batch_size: (step + 1) * batch_size] = np.squeeze(lbl)
        preds[step * batch_size: (step + 1) * batch_size] = np.argmax(np.squeeze(logits), axis=1)

    # noinspection PyUnresolvedReferences
    acc = np.mean((preds == labels).astype(np.float32)) * 100.
    err = (100. - acc)
    print('>>> {} Accuracy (%): {:.4f}'.format(name, acc))
    print('>>> {} Error (%): {:.4f}\n'.format(name, err))
    value_summary({'{}/accuracy'.format(name): acc,
                   '{}/error'.format(name): err}, summary_writer, global_step)
    # Write scores
    if global_step:
        score_dir = pjoin(os.path.dirname(c.log_path), 'run_{:02d}___infer_{}'.format(c.run, name))
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
            with open(pjoin(score_dir, 'sparsity_values.csv'), 'a') as f:
                f.write('Global step,Total sparsity\r\n')
        with open(pjoin(score_dir, 'metric_scores.csv'), 'a') as f:
            f.write('{:d},{:.4f},{:.4f}\r\n'.format(global_step, acc, err))
        with open(pjoin(score_dir, 'sparsity_values.csv'), 'a') as f:
            f.write('{:d},{:.7f}\r\n'.format(global_step, sps))
    return acc


# def _eval_vqa(session, c, m, summary_writer, global_step):
#     """
#     Wrapper for running the validation loop of VQA.
#     """
#     name = m.name
#     split_size = c.split_sizes[name]
#     assert split_size % c.batch_size_eval == 0
#     num_batches = int(split_size / c.batch_size_eval)
#
#     results = []
#     print('\nEvaluating model...\n')
#     for step in tqdm(range(num_batches), desc='running eval', ncols=100):
#         preds, logits, question_id = session.run([m.predictions, m.logits, m.question_id])
#         # pprint(preds[:10])
#         for i in range(len(preds)):
#             answer = c.answer_list[int(preds[i])]
#             results.append(dict(answer=answer, question_id=int(question_id[i])))
#     res_json_filepath = pjoin(c.log_path, 'val_results_{:08d}.json'.format(global_step))
#     res_csv_filepath = pjoin(c.log_path, 'val_scores.csv'.format(global_step))
#     with open(res_json_filepath, 'w') as f:
#         json.dump(results, f)
#
#     # 'overall', 'perAnswerType'
#     acc = evaluate_answers(res_file=res_json_filepath)
#     acc_overall = acc['overall']
#     acc_per_ans = acc['perAnswerType']
#     print('>>> {} accuracy (overall): {:8.4f}\n'.format(name, acc_overall))
#     print('>>> {} accuracy (per answer type): {}\n'.format(name, sorted(acc_per_ans.items())))
#     value_summary({'{}/accuracy_overall'.format(name): acc_overall}, summary_writer, global_step)
#     csv_out = [global_step, acc_overall] + [v for k, v in sorted(acc_per_ans.items())]
#     csv_out = '\n' + ','.join(map(str, csv_out))
#     if not os.path.isfile(res_csv_filepath):
#         headers = 'Global step,Overall,' + ','.join(sorted(acc_per_ans))
#         csv_out = headers + csv_out
#     with open(res_csv_filepath, 'a') as f:
#         f.write(csv_out)
#     return acc


def try_to_train(train_fn, config, try_block=True):
    """Wrapper for the main training function."""
    # if config.resume_training:
    #     logger.info('Resuming training from checkpoint.')
    #     fp = os.path.join(config.log_path, 'config.pkl')
    #     config = cfg.load_config(fp)
    #     config.resume_training = True
    #     config.checkpoint_path = kwargs.pop('log_path')
    #     config.lr_end = kwargs.pop('lr_end')
    #     config.max_epoch = kwargs.pop('max_epoch')
    # else:
    #     config.save_config_to_file()
    if try_block:
        try:
            train_fn(config)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            error_log = sys.exc_info()
            traceback_extract = tb.format_list(tb.extract_tb(error_log[2]))
            if not os.path.exists(config.log_path):
                os.makedirs(config.log_path)
            err_msg = 'Error occured:\r\n\r\n%s\r\n' % str(error_log[0])
            err_msg += '%s\r\n%s\r\n\r\n' % (str(error_log[1]), str(error_log[2]))
            err_msg += '\r\n\r\nTraceback stack:\r\n\r\n'
            for entry in traceback_extract:
                err_msg += '%s\r\n' % str(entry)
            name = 'error__' + os.path.split(config.log_path)[1] + '.txt'
            with open(os.path.join(os.path.dirname(config.log_path), name), 'w') as f:
                f.write(err_msg)
            logger.warning('\nWARNING: An error has occurred.\n')
            logger.warning(err_msg)
            # tf.reset_default_graph()
    else:
        train_fn(config)
