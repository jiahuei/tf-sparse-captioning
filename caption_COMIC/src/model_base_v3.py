#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:43:38 2017

@author: jiahuei
"""
import os
import logging
import math
import numpy as np
import tensorflow as tf
from functools import partial
from itertools import chain
from pprint import pprint
from tensorflow.python.layers.core import Dense
from common.natural_sort import natural_keys
from common.nets import nets_factory
from common.mask_prune import pruning
from common.mask_prune import masked_layer
from common.mask_prune import sparse_layer
import common.ops_v1 as ops
import common.ops_rnn_v2 as rops
import common.rnn_cells_v1 as rnn

logger = logging.getLogger(__name__)
_shape = ops.shape
pjoin = os.path.join


class ModelBase(object):
    """
    Base for model implementations.
    """

    def __init__(self, config, mode):
        self._config = c = config
        self.mode = mode
        self.batch_ops = self._dtype = self.reuse = None

        assert c.token_type in ['radix', 'word', 'char']
        if c.token_type == 'radix':
            self._vocab_size = c.radix_base + 2
        else:
            self._vocab_size = len(c.itow)

        self._mask_params = {}
        if c.supermask_type:
            self._dense_layer = masked_layer.MaskedDense
            self._mask_params = dict(mask_type=c.supermask_type,
                                     mask_init_value=c.supermask_init_value,
                                     mask_bern_sample=self.is_training, )
        elif hasattr(c, 'is_sparse') and c.is_sparse:
            raise NotImplementedError
            self._dense_layer = sparse_layer.SparseDense
        else:
            self._dense_layer = Dense

    #####################
    ### Input helpers ###

    def get_seq_mask_and_len(self, token_ids, time_axis=1):
        """
        Returns bool sequence mask and int32 sequence length tensors.
        :param token_ids:
        :param time_axis:
        :return:
        """
        c = self._config
        _PAD = '<PAD>'
        assert _PAD in c.wtoi
        pad_id = tf.cast(c.wtoi[_PAD], token_ids.dtype)
        mask = tf.cast(tf.not_equal(token_ids, pad_id), tf.float32)
        seq_len = tf.reduce_sum(mask, axis=time_axis)
        seq_len = tf.cast(seq_len, tf.int32)
        return mask, seq_len

    def _process_inputs(self):
        """
        Generates the necessary inputs, targets, masks.
        """
        c = self._config
        encoder_inputs, decoder_inputs = self.batch_ops
        self.encoder_inputs = dict(images=encoder_inputs)

        if self.is_inference:
            dec_seq_inputs = dec_seq_targets = dec_seq_masks = dec_seq_lens = None
        else:
            dec_seq_masks, dec_seq_lens = self.get_seq_mask_and_len(decoder_inputs[:, 1:])  # Exclude <GO>
            # Clip padding values at zero
            decoder_inputs = tf.maximum(decoder_inputs, 0)
            dec_seq_inputs = decoder_inputs[:, :-1]
            dec_seq_targets = decoder_inputs[:, 1:]
        self.decoder_inputs = dict(inputs=dec_seq_inputs, targets=dec_seq_targets,
                                   masks=dec_seq_masks, seq_lens=dec_seq_lens)

    def _build_word_projections(self):
        """Helper to update word embedding and output projection variables."""
        c = self._config
        if self.is_training and hasattr(c, 'use_glove_embeddings') and c.use_glove_embeddings:
            initialiser = self._get_glove_embedding_init()
            shape = None
        else:
            initialiser = None
            shape = [self._vocab_size, c.rnn_word_size]
        kwargs = dict(name='embedding_map',
                      initializer=initialiser,
                      shape=shape,
                      dtype=tf.float32,
                      trainable=True)
        if c.token_type == 'word':
            with tf.device('/cpu:0'):
                self._word_embed_map = tf.get_variable(**kwargs)
        else:
            self._word_embed_map = tf.get_variable(**kwargs)

        if c.supermask_type:
            self._word_embed_map, _ = masked_layer.generate_masks(kernel=self._word_embed_map,
                                                                  bias=None,
                                                                  mask_shape=None,
                                                                  dtype=None,
                                                                  **self._mask_params)
        return self._word_embed_map

    def _get_glove_embedding_init(self):
        c = self._config
        with open(c.glove_filepath, encoding='utf-8') as f:
            entries = f.readlines()
        emb_dim = len(entries[0].split(' ')) - 1
        if c.rnn_word_size != emb_dim:
            raise ValueError('Dimension mismatch: `rnn_word_size` is {} but GloVe has dim {}'.format(
                c.rnn_word_size, emb_dim))
        logger.info('Initialising word embedding matrix with {}-dim GloVe vectors.'.format(emb_dim))
        word2emb = {}
        for entry in entries:
            vals = entry.split(' ')
            word = vals[0]
            vals = list(map(float, vals[1:]))
            word2emb[word] = np.array(vals)
        loaded = 0
        word_embed_map = np.zeros((self._vocab_size, emb_dim), dtype=np.float32)
        for idx, word in c.itow.items():
            idx = int(idx)
            if idx < 0 or word not in word2emb:
                continue  # <PAD> has index of -1
            word_embed_map[idx] = word2emb[word]
            loaded += 1
        logger.debug('Percentage of vocab loaded from GloVe: {:4.1f} %'.format(loaded / self._vocab_size * 100.))
        return word_embed_map

    def _get_embedding_var_or_fn(self, tokens):
        c = self._config
        token_type = c.token_type

        if token_type == 'word':
            if self.is_inference:
                embeddings = self._word_embed_map
            else:
                with tf.device('/cpu:0'):
                    embeds = tf.nn.embedding_lookup(self._word_embed_map, tokens)  # (batch_size, seq_len, word_size)
                embeddings = tf.transpose(embeds, [1, 0, 2])  # (seq_len, batch_size, word_size)
        else:
            def _embed_fn(ids):
                return tf.gather(params=self._word_embed_map, indices=ids, axis=0)

            if self.is_inference:
                embeddings = _embed_fn
            else:
                embeds = _embed_fn(tokens)
                embeddings = tf.transpose(embeds, [1, 0, 2])  # (max_time, batch_size, word_size)
        return embeddings

    def _get_attention_mech(self):
        c = self._config
        align = c.attn_alignment_method
        prob = c.attn_probability_fn
        attn_size = c.attn_size

        if align == 'add_LN':
            att_mech = rops.MultiHeadAddLN
        elif align == 'add':
            att_mech = rops.MultiHeadAdd
        elif align == 'dot':
            att_mech = rops.MultiHeadDot
        else:
            raise ValueError('Invalid alignment method.')

        if prob == 'softmax':
            prob_fn = tf.nn.softmax
        elif prob == 'sigmoid':
            prob_fn = self._signorm
        else:
            raise ValueError('Invalid alignment method.')

        return partial(att_mech,
                       num_units=attn_size,
                       fm_projection=c.cnn_fm_projection,
                       num_heads=c.attn_num_heads,
                       probability_fn=prob_fn,
                       **self._mask_params)

    # TODO: Bookmark
    #############################################
    # Encoder & Decoder functions               #
    #############################################

    ###############
    ### Encoder ###

    def _encoder(self, flatten_spatial_dims=True):
        """
        Encoder CNN.
        
        Builds image CNN model specified by config.cnn_name.
        
        Setups and returns the following:
        self.encoder_final_state: A tensor of shape [batch, image_embed_size].
        self.encoder_outputs: A list of feature maps specified by
            config.cnn_fm_attention, each of shape
            [batch, map_height * map_width, channels].
        """
        c = self._config
        _mask_params = self._mask_params if 'masked' in c.cnn_name else {}
        # Select the CNN
        with tf.variable_scope('cnn'):
            cnn_fn = nets_factory.get_network_fn(c.cnn_name,
                                                 num_classes=None,
                                                 weight_decay=0.0,
                                                 is_training=False)
            net, end_points = cnn_fn(self.encoder_inputs['images'], global_pool=True, **_mask_params)

        # Produce image embeddings
        if c.legacy:
            net = ops.layer_norm_activate(scope='LN_tanh',
                                          inputs=tf.squeeze(net, axis=[1, 2]),
                                          activation_fn=tf.nn.tanh,
                                          begin_norm_axis=1)
            self.encoder_final_state = ops.linear(scope='im_embed',
                                                  inputs=net,
                                                  output_dim=1024,
                                                  bias_init=None,
                                                  activation_fn=None)
        else:
            self.encoder_final_state = tf.squeeze(net, axis=[1, 2])

        # Feature maps
        # Reshape CNN feature map for RNNs
        # Must have fully defined inner dims
        if c.cnn_fm_attention in end_points:
            cnn_fm = end_points[c.cnn_fm_attention]
        else:
            _err = '\n'.join(['{} --- {}'.format(k, v.shape.as_list())
                              for k, v in sorted(end_points.items(),
                                                 key=lambda x: natural_keys(x[0]))])
            _err = 'Invalid feature map name: `{}`. Available choices: \n{}'.format(
                c.cnn_fm_attention, _err)
            raise ValueError(_err)
        fm_ds = tf.shape(cnn_fm)  # (n, h, w, c)
        fm_ss = _shape(cnn_fm)
        fm_s = tf.stack([fm_ss[0], fm_ds[1] * fm_ds[2], fm_ss[3]], axis=0)
        cnn_fm = tf.reshape(cnn_fm, fm_s)
        self.encoder_outputs = cnn_fm
        return self.encoder_final_state, cnn_fm

    def _encoder_rnn(self):
        """
        RNN encoder for testing SNIP on MNIST.
        """
        c = self._config
        cell = self._get_rnn_cell(c.rnn_size, input_size=_shape(self.encoder_inputs['input_seq'])[-1])

        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        # noinspection PyUnresolvedReferences
        outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=self.encoder_inputs['input_seq'],
                                                 sequence_length=None,
                                                 initial_state=None,
                                                 dtype=self._dtype,
                                                 parallel_iterations=None,
                                                 swap_memory=True,
                                                 time_major=True)

        if 'LSTM' in c.rnn_name:
            final_out = final_state.h
        else:
            final_out = final_state
        self.encoder_outputs = outputs
        self.encoder_final_state = final_out
        return final_out, outputs

    ###############
    ### Decoder ###

    def _decoder_rnn(self):
        """
        RNN Decoder.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        enc_final_state = self.encoder_final_state
        enc_outputs = self.encoder_outputs

        align = c.attn_alignment_method
        prob = c.attn_probability_fn
        rnn_size = c.rnn_size
        attn_size = c.attn_size
        att_keep_prob = c.attn_keep_prob if self.is_training else 1.0
        batch_size = _shape(enc_final_state)[0]
        beam_search = (self.is_inference and c.infer_beam_size > 1)

        if beam_search:
            beam_size = c.infer_beam_size
            # Tile the batch dimension in preparation for Beam Search
            enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, beam_size)
            enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, beam_size)

        # if align == 'add_LN':
        #     att_mech = rops.MultiHeadAddLN
        # elif align == 'add':
        #     att_mech = rops.MultiHeadAdd
        # elif align == 'dot':
        #     att_mech = rops.MultiHeadDot
        # else:
        #     raise ValueError('Invalid alignment method.')
        #
        # if prob == 'softmax':
        #     prob_fn = tf.nn.softmax
        # elif prob == 'sigmoid':
        #     prob_fn = self._signorm
        # else:
        #     raise ValueError('Invalid alignment method.')

        if c.cnn_fm_projection is None and c.attn_context_layer is False:
            rnn_input_size = _shape(self.encoder_outputs)[-1]
        else:
            rnn_input_size = c.attn_size
        rnn_input_size += c.rnn_word_size

        with tf.variable_scope('rnn_decoder'):
            cell = self._get_rnn_cell(rnn_size, input_size=rnn_input_size)
            rnn_init = self._get_rnn_init(enc_final_state, cell, input_size=rnn_input_size)
            # cnn_attention = att_mech(num_units=attn_size,
            #                          feature_map=enc_outputs,
            #                          fm_projection=c.cnn_fm_projection,
            #                          num_heads=c.attn_num_heads,
            #                          memory_sequence_length=None,
            #                          probability_fn=prob_fn,
            #                          **self._mask_params)
            cnn_attention = self._get_attention_mech()(feature_map=enc_outputs, memory_sequence_length=None)
            attention_cell = rops.MultiHeadAttentionWrapperV3(context_layer=c.attn_context_layer,
                                                              alignments_keep_prob=att_keep_prob,
                                                              cell=cell,
                                                              attention_mechanism=cnn_attention,
                                                              attention_layer_size=None,
                                                              alignment_history=True,
                                                              cell_input_fn=None,
                                                              output_attention=False,
                                                              initial_cell_state=rnn_init,
                                                              **self._mask_params)

            self._build_word_projections()
            logger.debug('Building separate embedding matrix.')
            self.decoder_output_layer = self._dense_layer(
                units=self._vocab_size, name='output_projection', **self._mask_params)
            self.decoder_output_layer.build(rnn_size)
            embeddings = self._get_embedding_var_or_fn(self.decoder_inputs['inputs'])
            rnn_raw_outputs = self._rnn_dynamic_decoder(attention_cell,
                                                        embeddings,
                                                        self.decoder_output_layer)

        with tf.name_scope('post_processing'):
            logits, output_ids, attn_maps = self._decoder_post_process(rnn_raw_outputs, top_beam=True)
        self.dec_preds = output_ids
        self.dec_logits = logits
        self.dec_attn_maps = attn_maps
        return logits, output_ids, attn_maps

    def _decoder_rnn_scst(self, beam_size=0):
        """
        RNN Decoder for SCST training.
        
        NOTE: This function operates in batch-major mode.
        """
        c = self._config
        enc_final_state = self.encoder_final_state
        enc_outputs = self.encoder_outputs

        align = c.attn_alignment_method
        prob = c.attn_probability_fn
        rnn_size = c.rnn_size
        attn_size = c.attn_size
        att_keep_prob = c.attn_keep_prob if self.is_training else 1.0
        batch_size = _shape(enc_final_state)[0]

        sample = False
        if not self.is_training:
            if beam_size == 0:
                sample = True
            else:
                # Prepare beam search to sample candidates
                c.batch_size_infer = batch_size
                c.infer_beam_size = beam_size
                c.infer_max_length = 20
                c.infer_length_penalty_weight = 0
                # Tile the batch dimension in preparation for Beam Search
                enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, beam_size)
                enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, beam_size)

        # if align == 'add_LN':
        #     att_mech = rops.MultiHeadAddLN
        # elif align == 'add':
        #     att_mech = rops.MultiHeadAdd
        # elif align == 'dot':
        #     att_mech = rops.MultiHeadDot
        # else:
        #     raise ValueError('Invalid alignment method.')
        #
        # if prob == 'softmax':
        #     prob_fn = tf.nn.softmax
        # elif prob == 'sigmoid':
        #     prob_fn = self._signorm
        # else:
        #     raise ValueError('Invalid alignment method.')

        if c.cnn_fm_projection is None and c.attn_context_layer is False:
            rnn_input_size = _shape(self.encoder_outputs)[-1]
        else:
            rnn_input_size = c.attn_size
        rnn_input_size += c.rnn_word_size

        with tf.variable_scope('rnn_decoder'):
            cell = self._get_rnn_cell(rnn_size, input_size=rnn_input_size)
            rnn_init = self._get_rnn_init(enc_final_state, cell, input_size=rnn_input_size)
            # cnn_attention = att_mech(num_units=attn_size,
            #                          feature_map=enc_outputs,
            #                          fm_projection=c.cnn_fm_projection,
            #                          num_heads=c.attn_num_heads,
            #                          memory_sequence_length=None,
            #                          probability_fn=prob_fn,
            #                          **self._mask_params)
            cnn_attention = self._get_attention_mech()(feature_map=enc_outputs, memory_sequence_length=None)
            attention_cell = rops.MultiHeadAttentionWrapperV3(context_layer=c.attn_context_layer,
                                                              alignments_keep_prob=att_keep_prob,
                                                              cell=cell,
                                                              attention_mechanism=cnn_attention,
                                                              attention_layer_size=None,
                                                              alignment_history=True,
                                                              cell_input_fn=None,
                                                              output_attention=False,
                                                              initial_cell_state=rnn_init,
                                                              **self._mask_params)

            self._build_word_projections()
            logger.debug('Building separate embedding matrix.')
            self.decoder_output_layer = self._dense_layer(
                units=self._vocab_size, name='output_projection', **self._mask_params)
            self.decoder_output_layer.build(rnn_size)
            embeddings = self._get_embedding_var_or_fn(self.decoder_inputs['inputs'])
            rnn_raw_outputs = self._rnn_dynamic_decoder(attention_cell,
                                                        embeddings,
                                                        self.decoder_output_layer,
                                                        sample=sample)

        with tf.name_scope('post_processing'):
            logits, output_ids, attn_maps = self._decoder_post_process(rnn_raw_outputs, top_beam=False)
        self.dec_preds = output_ids
        self.dec_logits = logits
        self.dec_attn_maps = attn_maps
        return logits, output_ids, attn_maps

    def _decoder_post_process(self, rnn_raw_outputs, top_beam=True):
        c = self._config
        beam_search = (self.is_inference and len(_shape(rnn_raw_outputs[0])) > 2)

        if beam_search:
            predicted_ids, scores, dec_states = rnn_raw_outputs  # (time, batch_size, beam_size)
            if top_beam:
                # Beams are sorted from best to worst according to prob
                top_sequence = predicted_ids[:, :, 0]
                top_score = scores[:, :, 0]
                # (batch_size, seq_len)
                output_ids = tf.transpose(top_sequence, [1, 0])
                logits = tf.transpose(top_score, [1, 0])
            else:
                output_ids = tf.transpose(predicted_ids, [2, 1, 0])  # (beam_size, batch_size, time)
                logits = tf.transpose(scores, [2, 1, 0])
        else:
            output_ids, logits, dec_states = rnn_raw_outputs
            # (batch_size, seq_len, softmax_size)
            logits = tf.transpose(logits, [1, 0, 2])
            output_ids = tf.transpose(output_ids, [1, 0])

        ## Attention Map ##
        attn_map = dec_states.alignment_history
        # (seq_len, batch * beam, num_heads * fm_size)
        if beam_search:
            assert not isinstance(attn_map, tf.TensorArray)
            beam_size = c.infer_beam_size
            # Select top beam
            map_s = tf.shape(attn_map)
            map_s = tf.stack([map_s[0], -1, beam_size, map_s[2]], axis=0)
            attn_map = tf.reshape(attn_map, map_s)
            attn_map = attn_map[:, :, 0, :]
            # (seq_len, batch, fm_size)
        else:
            attn_map = attn_map.stack()
        # Retrieve the attention maps (seq_len, batch, num_heads * fm_size)
        map_s = tf.shape(attn_map)
        map_s = tf.stack([map_s[0], map_s[1], c.attn_num_heads, -1], axis=0)
        attn_map = tf.reshape(attn_map, map_s)  # (seq_len, batch, num_heads, fm_size)
        attn_map = tf.transpose(attn_map, [1, 2, 0, 3])  # (batch, num_heads, seq_len, fm_size)

        return logits, output_ids, attn_map

    # TODO: Bookmark
    #############################################
    # Loss & Training & Restore functions       #
    #############################################

    #############################
    ### Loss fns & Optimisers ###

    def _compute_caption_loss(self, scst=False):
        """
        Calculates the average log-perplexity per word, and also the
        doubly stochastic loss of attention map.
        """
        if self.is_inference:
            return None
        c = self._config

        with tf.name_scope('loss'):
            ### Sequence / Reconstruction loss ###
            with tf.name_scope('decoder'):
                if not scst:
                    dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                        logits=self.dec_logits,
                        targets=self.decoder_inputs['targets'],
                        weights=self.decoder_inputs['masks'])
                else:
                    # if c.supermask_type:
                    #     raise ValueError('SCST not compatible with Supermask')
                    dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                        logits=self.dec_logits,
                        targets=self.decoder_inputs['targets'],
                        weights=self.decoder_inputs['masks'],
                        average_across_batch=False)
                    # noinspection PyUnresolvedReferences
                    dec_log_ppl = tf.reduce_mean(dec_log_ppl * self.rewards)

                tf.summary.scalar('loss', dec_log_ppl)
                tf.summary.scalar('perplexity', tf.exp(dec_log_ppl))
                self.model_loss = self.snip_loss = dec_log_ppl
            return self.model_loss

    def _maybe_add_train_op(self):
        """
        Calculates the average log-perplexity per word, and also the
        doubly stochastic loss of attention map.
        """
        if not self.is_training:
            return None
        c = self._config

        with tf.name_scope('loss'):
            # Attention map doubly stochastic loss
            map_loss = 0.
            if c.attn_map_loss_scale > 0:
                logger.warning('Attention map loss is discouraged.')
                # Maps (batch, num_heads, seq_len, fm_size), Masks (batch, seq_len)
                with tf.name_scope('attention_map'):
                    masks = tf.expand_dims(tf.expand_dims(self.decoder_inputs['masks'], axis=1), axis=3)
                    # Sum along time and head dimensions into (batch, fm_size)
                    flat_cnn_maps = tf.reduce_sum(tf.multiply(masks, self.dec_attn_maps), axis=[1, 2])
                    map_loss = tf.squared_difference(1.0, flat_cnn_maps)
                    map_loss = tf.reduce_mean(map_loss)
                    tf.summary.scalar('loss', map_loss)
                    map_loss = tf.multiply(map_loss, c.attn_map_loss_scale)
                    tf.summary.scalar('loss_weighted', map_loss)

            # Mask losses
            mask_loss = 0.
            if c.supermask_type in masked_layer.SUPER_MASKS \
                    and c.supermask_sparsity_weight > 0 \
                    and c.supermask_lr_start > 0:
                with tf.name_scope('supermask'):
                    mask_loss = pruning.sparsity_loss(sparsity_target=c.supermask_sparsity_target,
                                                      loss_type=c.supermask_sparsity_loss_fn,
                                                      exclude_scopes=c.prune_freeze_scopes)
                    tf.summary.scalar('loss', mask_loss)
                    mask_wg = c.supermask_sparsity_weight
                    if c.supermask_loss_anneal:
                        mask_wg *= (1. - self._anneal_rate)
                    mask_loss = tf.multiply(mask_loss, mask_wg)
                    tf.summary.scalar('loss_weighted', mask_loss)

            # Retrieve and filter trainable variables
            if c.supermask_type == masked_layer.TRAIN_MASK:
                raise NotImplementedError
                tvars = tvars_mask = tf.get_collection(masked_layer.MASK_COLLECTION)
            else:
                # if c.supermask_type in masked_layer.MAG_PRUNE_MASKS + masked_layer.MASK_PRUNE:
                # if c.supermask_type in masked_layer.MAG_PRUNE_MASKS:
                #     tvars_mask = []
                # else:
                #     tvars_mask = tf.get_collection(masked_layer.MASK_COLLECTION)
                tvars = self._get_trainable_vars()

            tvars_cnn = tf.contrib.framework.filter_variables(
                var_list=tvars,
                include_patterns=['Model/encoder/cnn'],
                exclude_patterns=['mask'],
                reg_search=True)
            tvars_dec = tf.contrib.framework.filter_variables(
                var_list=tvars,
                include_patterns=['Model'],
                exclude_patterns=['Model/encoder/cnn', 'mask'],
                reg_search=True)
            tvars_mask = tf.contrib.framework.filter_variables(
                var_list=tvars,
                include_patterns=['mask'],
                exclude_patterns=None,
                reg_search=True)
            assert len(tvars) == len(tvars_cnn + tvars_dec + tvars_mask)

            # Add losses
            reg_loss = self._loss_regularisation(tvars_cnn + tvars_dec)
            loss = self.model_loss + reg_loss + mask_loss + map_loss
            tf.summary.scalar('total_loss', loss)

        # Training op for captioning model
        with tf.variable_scope('optimise/caption'):
            if c.cnn_grad_multiplier != 1.0:
                logger.debug('Using gradient multipliers.')
                multipliers = dict(
                    list(zip(tvars_cnn, [c.cnn_grad_multiplier] * len(tvars_cnn))) +
                    list(zip(tvars_dec, [1.0] * len(tvars_dec))))
            else:
                multipliers = None

            if c.supermask_type in masked_layer.SUPER_MASKS:
                multipliers = None
                grad_vars = ops.compute_gradients(total_loss=loss,
                                                  variables_to_train=tvars,
                                                  clip_gradient_norm=0,
                                                  summarize_gradients=c.add_grad_summaries,
                                                  gradient_multipliers=multipliers)

                def _clip_fn(x):
                    if c.clip_gradient_norm > 0:
                        return tf.clip_by_norm(x, c.clip_gradient_norm)
                    else:
                        return x

                # Filter out masks
                grad_vars_mask = []
                grad_vars_model = []
                for g, v in grad_vars:
                    if 'mask' in v.op.name:
                        grad_vars_mask.append((g, v))
                    else:
                        # Only clip gradients of model variables
                        grad_vars_model.append((_clip_fn(g), v))
                del grad_vars
                assert len(grad_vars_model + grad_vars_mask) == len(tvars)
                print('----------- -----------')
                print('----------- Model trainable variables -----------')
                pprint([_[1].op.name for _ in grad_vars_model])
                print('----------- Mask trainable variables -----------')
                pprint([_[1].op.name for _ in grad_vars_mask])
                print('----------- -----------')

                optimisers = [self._get_optimiser(self.lr, momentum=None)]
                grads_and_vars = [grad_vars_model]
                if c.supermask_lr_start > 0 and len(grad_vars_mask) > 0:
                    logger.debug('Using separate optimisers.')
                    optimisers += [self._get_optimiser(c.supermask_lr_start, momentum=None)]
                    grads_and_vars += [grad_vars_mask]
                else:
                    logger.debug('Freezing masks.')
                self.model_loss = ops.create_train_op(total_loss=loss,
                                                      optimizers=optimisers,
                                                      grads_and_vars=grads_and_vars,
                                                      global_step=self.global_step,
                                                      check_numerics=True)
                self.final_trainable_variables = [v for g, v in chain.from_iterable(grads_and_vars)]
            else:
                # https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/slim/python/slim/learning.py
                # https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/training/python/training/training.py#L370
                self.model_loss = tf.contrib.slim.learning.create_train_op(
                    loss,
                    self._get_optimiser(self.lr, momentum=None),
                    global_step=self.global_step,
                    variables_to_train=tvars,
                    clip_gradient_norm=c.clip_gradient_norm,
                    summarize_gradients=c.add_grad_summaries,
                    gradient_multipliers=multipliers)
                self.final_trainable_variables = tvars

            # with tf.control_dependencies([train_cap]):
            #     self.model_loss = tf.identity(self.model_loss)
        return self.model_loss

    def _maybe_add_magnitude_pruning_ops(self):
        c = self._config
        if not self.is_training or c.supermask_type not in masked_layer.MAG_PRUNE_MASKS + [masked_layer.LOTTERY]:
            return None
        self.all_pruned_weights = pruning.get_weights(exclude_scopes=None)
        self.all_pruning_masks = pruning.get_masks(exclude_scopes=None)[1]
        masks, _ = pruning.get_masks(exclude_scopes=c.prune_freeze_scopes)
        self.masks = masks
        print('----------- Mask non-trainable variables -----------')
        pprint([_.op.name for _ in masks])

        if c.supermask_type in masked_layer.MAG_HARD + [masked_layer.LOTTERY]:
            logger.info('Pruning ({}): Setting up mask assign ops.'.format(c.supermask_type))
            self.mask_assign_ops = pruning.get_mask_assign_ops(
                mask_type=c.supermask_type,
                sparsity_target=c.supermask_sparsity_target,
                exclude_scopes=c.prune_freeze_scopes)
        elif c.supermask_type == masked_layer.SNIP:
            logger.info('Pruning (SNIP): Setting up mask assign ops.')
            self.mask_assign_ops, self.accum_saliency = pruning.get_mask_assign_ops(
                mask_type=c.supermask_type,
                sparsity_target=c.supermask_sparsity_target,
                exclude_scopes=c.prune_freeze_scopes,
                loss=self.snip_loss)
        else:
            logger.info('Pruning ({}): Setting up conditional mask assign ops.'.format(c.supermask_type))
            # Gradual pruning
            assert c.supermask_type in masked_layer.MAG_ANNEAL
            # if c.train_mode != 'decoder':
            #     return None
            if c.supermask_type == masked_layer.MAG_GRAD_UNIFORM:
                prune_scheme = masked_layer.MAG_UNIFORM
            else:
                prune_scheme = masked_layer.MAG_BLIND
            global_step = tf.subtract(self.global_step, 1)  # training op will increment global_step by 1
            prune_freq = 1000
            prune_start = int((1 / c.max_epoch) * c.max_step)  # start of 2nd epoch
            n = int((0.50 * c.max_step - prune_start) / prune_freq)
            # prune_start = int(0.25 * c.max_step)
            # n = int((0.75 * c.max_step - prune_start) / prune_freq)
            pruning_end = prune_start + prune_freq * n
            mask_update_op = pruning.conditional_mask_update_op(
                exclude_scopes=c.prune_freeze_scopes,
                pruning_scheme=prune_scheme,
                global_step=global_step,
                initial_sparsity=0.,
                final_sparsity=c.supermask_sparsity_target,
                pruning_start_step=prune_start,
                pruning_end_step=pruning_end,
                prune_frequency=prune_freq)
            with tf.control_dependencies([mask_update_op]):
                self.model_loss = tf.identity(self.model_loss)

        # For learnable masks, summary ops are added in `sparsity loss`
        if len(masks) > 0:
            pruning.mask_sparsity_summaries(masks, [m.op.name for m in masks])

    def maybe_run_assign_masks(self, session):
        c = self._config
        if c.supermask_type not in masked_layer.MAG_PRUNE_MASKS + [masked_layer.LOTTERY]:
            logger.info('Pruning ({}): Invalid mask type, skipping mask assign.'.format(c.supermask_type))
            return None, None, None
        if c.supermask_type not in masked_layer.MAG_ANNEAL:
            # Annealed magnitude pruning do not prune at the beginning
            if c.supermask_type == masked_layer.SNIP:
                accum_batch = 1
                logger.info('Pruning (SNIP): Accumulating saliency using {} batch(es)'.format(accum_batch))
                for i in range(accum_batch):
                    session.run(self.accum_saliency)
            # Run pruning ops
            logger.info('Pruning ({}): Assigning masks.'.format(c.supermask_type))
            session.run(self.mask_assign_ops)
        masks = session.run(self.masks)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Pruning ({}):\nMasks: {}'.format(c.supermask_type, masks))
        total_size = sum([np.prod(m.shape) for m in masks])
        total_nnz = np.sum([np.sum(m) for m in masks])
        logger.info('Pruning ({}): Sparsity level: {:4.3f}'.format(
            c.supermask_type, 1 - (total_nnz / total_size))
        )
        return masks, total_size, total_nnz

    def _loss_regularisation(self, var_list):
        """ Add L2 regularisation. """
        c = self._config
        loss = .0
        if c.l2_decay > 0:
            with tf.name_scope('regularisation'):
                for var in var_list:
                    loss += ops.l2_regulariser(var, c.l2_decay)
        tf.summary.scalar('regularisation_loss', loss)
        return loss

    #################
    ### Restoring ###

    def restore_model(self, session, saver, lr):
        """
        Helper function to restore model variables.
        """
        c = self._config
        print('\n')

        if not c.checkpoint_path:
            logger.info('Training entire model from scratch.')
        else:
            if os.path.isfile(c.checkpoint_path + '.index') \
                    or os.path.isfile(c.checkpoint_path):  # V2 & V1 checkpoint
                checkpoint_path = c.checkpoint_path
            else:
                checkpoint_path = tf.train.latest_checkpoint(c.checkpoint_path)

            ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
            ckpt_vars_namelist = set(ckpt_reader.get_variable_to_shape_map().keys())
            # model_vars_namelist = set([v.op.name for v in self._get_trainable_vars()])
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model')
            if c.checkpoint_exclude_scopes != '':
                exc_scopes = [sc.strip() for sc in c.checkpoint_exclude_scopes.split(',')]
                var_list = tf.contrib.framework.filter_variables(var_list=var_list,
                                                                 include_patterns=['Model'],
                                                                 exclude_patterns=exc_scopes,
                                                                 reg_search=True)
                # model_vars_namelist = set([v.op.name for v in var_list])
            else:
                exc_scopes = None

            cnn_scope = 'Model/encoder/cnn/'
            not_slim_ckpt = any(['Model' in _ for _ in ckpt_vars_namelist])
            model_vars_namelist = set([v.op.name for v in var_list])
            # if model_vars_namelist.issubset(ckpt_vars_namelist):
            if not_slim_ckpt:
                if exc_scopes is None and c.resume_training:
                    logger.info('Resuming training from checkpoint: `{}`'.format(checkpoint_path))
                    # Restore whole model (resume training)
                    saver.restore(session, checkpoint_path)
                else:
                    logger.info('Restoring `Model` from checkpoint: `{}`'.format(checkpoint_path))
                    if model_vars_namelist.issubset(ckpt_vars_namelist):
                        # Restore whole model (fine-tune)
                        pass
                    else:
                        # # Prune CNN but load CNN weights without masks
                        # logger.debug('Restoring all model variables except CNN pruning masks.')
                        # var_list = tf.contrib.framework.filter_variables(var_list=var_list,
                        #                                                  include_patterns=None,
                        #                                                  exclude_patterns=[cnn_scope + '.+mask'],
                        #                                                  reg_search=True)
                        logger.debug('Some model variables are not found in checkpoint (probably pruning masks). '
                                     'Restoring as much as possible from checkpoint.')
                        var_list = [v for v in var_list if v.op.name in ckpt_vars_namelist]
                    logger.debug('Restoring variables: `{}`'.format([v.op.name for v in var_list]))
                    _saver = tf.train.Saver(var_list)
                    _saver.restore(session, checkpoint_path)
            else:
                # Restore CNN model
                logger.info('Restoring CNN model from checkpoint: `{}`'.format(checkpoint_path))
                if 'masked' in c.cnn_name:
                    logger.debug('Restoring masked CNN from slim checkpoint.')
                    exc_scopes = [] if exc_scopes is None else exc_scopes
                    exc_scopes += ['mask']
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cnn_scope)
                var_list = tf.contrib.framework.filter_variables(var_list=var_list,
                                                                 include_patterns=cnn_scope,
                                                                 exclude_patterns=exc_scopes,
                                                                 reg_search=True)  # Use re.search
                var_namelist = [v.op.name.replace(cnn_scope, '') for v in var_list]
                cnn_variables = dict(zip(var_namelist, var_list))
                cnn_saver = tf.train.Saver(cnn_variables)
                cnn_saver.restore(session, checkpoint_path)

        if self.is_training:
            if lr is None:
                lr = session.run(self.lr)
            else:
                self.update_lr(session, lr)
                session.run(self.lr)
        return lr

    # TODO: Bookmark
    #############################################
    # Model helper functions                    #
    #############################################

    ###################
    ### RNN helpers ###

    def _signorm(self, tn):
        with tf.variable_scope('sig_norm'):
            tn = tf.nn.sigmoid(tn)
            tn_sum = tf.reduce_sum(tn, axis=-1, keepdims=True)
            return tn / tn_sum

    def _get_rnn_cell(self, rnn_size, input_size):
        """Helper to select RNN cell(s)."""
        c = self._config
        use_masked_cell = c.supermask_type is not None
        cells = rnn.get_rnn_cell(name=c.rnn_name,
                                 num_units=rnn_size,
                                 reuse=self.reuse,
                                 use_fused_cell=True,
                                 use_masked_cell=use_masked_cell,
                                 use_sparse_cell=False,
                                 masked_cell_kwargs=self._mask_params)

        if c.rnn_layers > 1:
            raise ValueError('RNN layer > 1 not implemented.')
            # cells = tf.contrib.rnn.MultiRNNCell([cells] * self.config.num_layers)

        # Setup input and output dropouts
        input_keep = c.rnn_keep_in
        output_keep = c.rnn_keep_out
        if self.is_training and (input_keep < 1 or output_keep < 1):
            logger.debug('Training RNN using dropout.')
            cells = tf.contrib.rnn.DropoutWrapper(cells,
                                                  input_keep_prob=input_keep,
                                                  output_keep_prob=output_keep,
                                                  variational_recurrent=c.rnn_recurr_dropout,
                                                  input_size=input_size,
                                                  dtype=tf.float32)
        return cells

    def _get_rnn_init(self, sent_embeddings, cell, input_size=None):
        """
        Helper to generate initial state of RNN cell.
        """
        c = self._config
        rnn_init_method = c.rnn_init_method

        if rnn_init_method == 'project_hidden':
            if 'LSTM' in c.rnn_name:
                init_layer = self._dense_layer(units=cell.state_size[1],
                                               use_bias=False,
                                               activation=None,
                                               name='rnn_initial_state',
                                               **self._mask_params)
                init_state_h = init_layer(sent_embeddings)
                initial_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(init_state_h), init_state_h)

            else:
                init_layer = self._dense_layer(units=cell.state_size,
                                               use_bias=False,
                                               activation=None,
                                               name='rnn_initial_state',
                                               **self._mask_params)
                initial_state = init_layer(sent_embeddings)

        elif rnn_init_method == 'first_input':
            # Run the RNN cell once to initialise the hidden state
            name = 'rnn_init_input'
            with tf.name_scope(name):
                batch_size = _shape(sent_embeddings)[0]
                init_layer = self._dense_layer(units=input_size,
                                               use_bias=False,
                                               activation=None,
                                               name='{}/projection'.format(name),
                                               **self._mask_params)
                sent_embeddings = init_layer(sent_embeddings)
                initial_state = cell.zero_state(batch_size, dtype=tf.float32)
                _, initial_state = cell(sent_embeddings, initial_state)

        else:
            raise ValueError('Invalid RNN init method specified.')
        return initial_state

    def _rnn_dynamic_decoder(self,
                             cell,
                             embedding,
                             output_layer,
                             sample=False):

        c = self._config
        swap_memory = True
        if c.token_type == 'radix':
            start_id = tf.to_int32(c.radix_base)
            end_id = tf.to_int32(c.radix_base + 1)
        else:
            start_id = tf.to_int32(c.wtoi['<GO>'])
            end_id = tf.to_int32(c.wtoi['<EOS>'])

        if self.is_inference:
            maximum_iterations = c.infer_max_length
            if c.token_type == 'radix':
                max_word_len = len(ops.number_to_base(len(c.wtoi), c.radix_base))
                maximum_iterations *= max_word_len
            elif c.token_type == 'char':
                maximum_iterations *= 5
            beam_search = c.infer_beam_size > 1
            if sample:
                return rops.rnn_decoder_search(cell,
                                               embedding,
                                               output_layer,
                                               c.batch_size_infer,
                                               maximum_iterations,
                                               start_id,
                                               end_id,
                                               swap_memory,
                                               greedy_search=False)
            if beam_search:
                return rops.rnn_decoder_beam_search(cell,
                                                    embedding,
                                                    output_layer,
                                                    c.batch_size_infer,
                                                    c.infer_beam_size,
                                                    c.infer_length_penalty_weight,
                                                    maximum_iterations,
                                                    start_id,
                                                    end_id,
                                                    swap_memory)
            else:
                return rops.rnn_decoder_search(cell,
                                               embedding,
                                               output_layer,
                                               c.batch_size_infer,
                                               maximum_iterations,
                                               start_id,
                                               end_id,
                                               swap_memory,
                                               greedy_search=True)
        else:
            return rops.rnn_decoder_training(cell,
                                             embedding,
                                             output_layer,
                                             _shape(embedding)[1],
                                             self.decoder_inputs['seq_lens'],
                                             swap_memory)

    ########################
    ### Training helpers ###

    @property
    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == 'train'

    @property
    def is_inference(self):
        """Returns true if the model is built for training mode."""
        return self.mode == 'infer'

    def update_lr(self, session, lr_value):
        session.run(self._assign_lr, {self._new_lr: lr_value})

    def get_global_step(self, session):
        return session.run(self.global_step)

    def _create_gstep(self):
        """
        Helper to create global step variable.
        """
        # with tf.variable_scope('misc'):
        self.global_step = tf.get_variable(tf.GraphKeys.GLOBAL_STEP,
                                           shape=[],
                                           dtype=tf.int32,
                                           initializer=tf.zeros_initializer(),
                                           trainable=False,
                                           collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                        tf.GraphKeys.GLOBAL_STEP])
        self._new_step = tf.placeholder(tf.int32, None, 'new_global_step')
        self._assign_step = tf.assign(self.global_step, self._new_step)

    def _create_lr(self):
        """
        Helper to create learning rate variable.
        """
        # with tf.variable_scope('misc'):
        self.lr = tf.get_variable('learning_rate',
                                  shape=[],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        self._new_lr = tf.placeholder(tf.float32, None, 'new_lr')
        self._assign_lr = tf.assign(self.lr, self._new_lr)

    def _create_cosine_lr(self):
        """
        Helper to anneal learning rate following a cosine curve.
        """
        c = self._config
        self._create_lr()
        with tf.variable_scope('learning_rate'):
            step = tf.to_float(self.global_step / c.max_step)
            step = 1.0 + tf.cos(tf.minimum(1.0, step) * math.pi)
            self._anneal_rate = step / 2
            lr = (c.lr_start - c.lr_end) * self._anneal_rate + c.lr_end
        self.lr = lr

    def _get_initialiser(self):
        """Helper to select initialiser."""
        c = self._config
        if 'xavier' in c.initialiser:
            logger.debug('Using Xavier / Glorot {} initialiser.'.format(c.initialiser.split('_')[1]))
            uniform = 'uniform' in c.initialiser
            init = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                  mode='FAN_AVG',
                                                                  uniform=uniform,
                                                                  seed=None,
                                                                  dtype=self._dtype)
        elif c.initialiser == 'he':
            logger.debug('Using He / MSRA random normal initialiser.')
            init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                  mode='FAN_IN',
                                                                  uniform=False,
                                                                  seed=None,
                                                                  dtype=self._dtype)
        elif c.initialiser == 'truncated_normal':
            logger.debug('Using truncated random normal initialiser.')
            init = tf.truncated_normal_initializer(mean=0.0,
                                                   stddev=1.0,
                                                   seed=None,
                                                   dtype=self._dtype)
        else:
            raise ValueError('Invalid initialiser specified.')
        return init

    def _get_trainable_vars(self):
        """
        Helper to retrieve list of variables we want to train.
        """
        c = self._config
        tvars = tf.trainable_variables()

        if c.freeze_scopes:
            exc_scopes = c.freeze_scopes
            logger.info('Scopes freezed: {}'.format(exc_scopes))
        else:
            exc_scopes = []
        tvars = tf.contrib.framework.filter_variables(
            var_list=tvars,
            include_patterns=['Model'],
            exclude_patterns=exc_scopes + ['mask'],
            reg_search=True)
        if c.supermask_type:
            logger.info('Mask scopes freezed: {}'.format(c.prune_freeze_scopes))
            tvars_mask = tf.contrib.framework.filter_variables(
                var_list=tf.trainable_variables(),
                include_patterns=['mask'],
                exclude_patterns=c.prune_freeze_scopes,
                reg_search=True)
            tvars = tvars + tvars_mask
            # logger.debug('Mask variables added to list of trainable variables.')
        return tvars

    def _get_optimiser(self, lr, momentum=None):
        c = self._config
        opt_type = c.optimiser
        if opt_type == 'adam':
            # https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/training/adam.py
            if momentum is None:
                logger.debug('Using ADAM with default momentum values.')
                opt = tf.train.AdamOptimizer(learning_rate=lr,
                                             beta1=0.9, beta2=0.999,
                                             epsilon=c.adam_epsilon)
            else:
                logger.debug('Using ADAM with momentum: {}.'.format(momentum))
                opt = tf.train.AdamOptimizer(learning_rate=lr,
                                             beta1=momentum, beta2=0.999,
                                             epsilon=c.adam_epsilon)
        elif opt_type == 'sgd':
            if momentum is None:
                logger.debug('Using SGD default momentum values.')
                opt = tf.train.MomentumOptimizer(learning_rate=lr,
                                                 momentum=0.9,
                                                 use_nesterov=False)
            else:
                logger.debug('Using SGD with momentum: {}.'.format(momentum))
                opt = tf.train.MomentumOptimizer(learning_rate=lr,
                                                 momentum=momentum,
                                                 use_nesterov=False)
        else:
            raise ValueError('Unknown optimiser.')
        return opt


class ModelBaseNoCNN(ModelBase):

    # def restore_model(self, session, saver, lr):
    #     """
    #     Helper function to restore model variables.
    #     """
    #     c = self._config
    #     print('\n')
    #
    #     if not c.checkpoint_path:
    #         logger.info('Training entire model from scratch.')
    #     else:
    #         if os.path.isfile(c.checkpoint_path + '.index') \
    #                 or os.path.isfile(c.checkpoint_path):  # V2 & V1 checkpoint
    #             checkpoint_path = c.checkpoint_path
    #         else:
    #             checkpoint_path = tf.train.latest_checkpoint(c.checkpoint_path)
    #
    #         ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    #         ckpt_vars_namelist = set(ckpt_reader.get_variable_to_shape_map().keys())
    #         model_vars_namelist = set([v.op.name for v in self._get_trainable_vars()])
    #         var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model')
    #         if c.checkpoint_exclude_scopes != '':
    #             exc_scopes = [sc.strip()
    #                           for sc in c.checkpoint_exclude_scopes.split(',')]
    #             var_list = tf.contrib.framework.filter_variables(var_list=var_list,
    #                                                              include_patterns=['Model'],
    #                                                              exclude_patterns=exc_scopes,
    #                                                              reg_search=True)
    #             model_vars_namelist = set([v.op.name for v in var_list])
    #         else:
    #             exc_scopes = None
    #
    #         if model_vars_namelist.issubset(ckpt_vars_namelist):
    #             if exc_scopes is None and c.resume_training:
    #                 logger.info('Resuming training from checkpoint: `{}`'.format(checkpoint_path))
    #                 # Restore whole model (resume training)
    #                 saver.restore(session, checkpoint_path)
    #             else:
    #                 # Restore whole model (fine-tune)
    #                 logger.info('Restoring `Model` from checkpoint: `{}`'.format(checkpoint_path))
    #                 _saver = tf.train.Saver(var_list)
    #                 _saver.restore(session, checkpoint_path)
    #
    #     if self.is_training:
    #         if lr is None:
    #             lr = session.run(self.lr)
    #         else:
    #             self.update_lr(session, lr)
    #             session.run(self.lr)
    #     return lr

    def _process_inputs(self):
        """
        Generates the necessary inputs, targets, masks.
        """
        c = self._config
        self._enc_inputs = self.batch_ops[0]
        self._enc_seq_masks = tf.sign(tf.to_float(self._enc_inputs + 1))
        self._enc_seq_lens = tf.reduce_sum(self._enc_seq_masks, axis=1)
        self._enc_seq_lens = tf.to_int32(self._enc_seq_lens)
        self._enc_inputs = tf.maximum(self._enc_inputs, 0)

        if self.is_inference:
            self._dec_seq_lens = None
        else:
            _dec_seq = self.batch_ops[1]  # Decoder sentences (word IDs)

            self._dec_seq_masks = tf.sign(tf.to_float(_dec_seq[:, 1:] + 1))  # Exclude <GO>
            self._dec_seq_lens = tf.reduce_sum(self._dec_seq_masks, axis=1)
            self._dec_seq_lens = tf.to_int32(self._dec_seq_lens)
            # Clip padding values at zero
            _dec_seq = tf.maximum(_dec_seq, 0)
            self._dec_seq_inputs = _dec_seq[:, :-1]
            self._dec_seq_targets = _dec_seq[:, 1:]


class VQABase(ModelBase):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.logits = self.predictions = None

    def _process_inputs(self):
        """
        Generates the necessary inputs, targets, masks.
        """
        c = self._config
        if self.is_inference:
            images, questions, question_id = self.batch_ops
            answers = None
        else:
            images, questions, question_id, answers = self.batch_ops
            print('`images` shape: {}'.format(_shape(images)))
            print('`questions` shape: {}'.format(_shape(questions)))
            print('`answers` shape: {}'.format(_shape(answers)))
            print('`question_id` shape: {}'.format(_shape(question_id)))

        # Mask short questions with zeroes
        questions_masks, questions_seq_lens = self.get_seq_mask_and_len(questions)
        questions_masks = tf.expand_dims(tf.transpose(questions_masks, [1, 0]), 2)
        self._build_word_projections()
        questions_embed = self._get_embedding_var_or_fn(tokens=tf.maximum(questions, 0))
        questions_embed = tf.multiply(questions_embed, questions_masks)
        assert _shape(questions_embed)[0] == c.question_max_len
        self.encoder_inputs = dict(images=images, input_seq=questions_embed)
        self.answers = answers
        self.question_id = question_id

    def _maybe_dropout(self, tensor, keep_prob):
        if keep_prob < 1. and self.is_training:
            tensor = tf.contrib.layers.dropout(inputs=tensor,
                                               keep_prob=keep_prob,
                                               noise_shape=None,
                                               is_training=True)
        return tensor

    def _gated_tanh(self, inputs, units, scope='gated_tanh'):
        c = self._config
        with tf.variable_scope(scope):
            inputs = self._maybe_dropout(inputs, c.keep_prob)
            y_layer = self._dense_layer(units=units, name='y', **self._mask_params)
            y = tf.nn.tanh(y_layer(inputs))
            g_layer = self._dense_layer(units=units, name='g', **self._mask_params)
            g = tf.nn.sigmoid(g_layer(inputs))
            return tf.multiply(y, g)

    def _fully_connected(self,
                         inputs,
                         units,
                         scope='fc',
                         activation_fn=tf.nn.relu,
                         use_batch_norm=False,
                         input_keep_prob=1.0):
        with tf.variable_scope(scope):
            inputs = self._maybe_dropout(inputs, input_keep_prob)
            if use_batch_norm:
                dense_activation = None
            else:
                dense_activation = activation_fn
            layer = self._dense_layer(
                units=units, activation=dense_activation, name='dense', **self._mask_params)
            outputs = layer(inputs)
            if use_batch_norm:
                outputs = tf.layers.batch_normalization(
                    outputs, training=self.is_training, renorm=True, fused=True, name='BN')
                if activation_fn is not None:
                    outputs = activation_fn(outputs)
            return outputs

    def _attention_module(self, query, keys, values):
        assert len(_shape(query)) == 3, '`query` must be rank 3 with shape (batch, 1, units)'
        assert len(_shape(keys)) == 3, '`keys` must be rank 3 with shape (batch, memory_size, units)'
        assert len(_shape(values)) == 3, '`values` must be rank 3 with shape (batch, memory_size, units)'
        c = self._config
        num_heads = c.attn_num_heads
        num_units = c.attn_size

        with tf.variable_scope('MLP'):
            keys = self._gated_tanh(keys, num_units, 'keys_gated_tanh')
            query = self._gated_tanh(query, num_units, 'query_gated_tanh')
            print('`keys` shape: {}'.format(_shape(keys)))
            print('`query` shape: {}'.format(_shape(query)))
            # Prepare query, keys, values
            if c.cnn_fm_projection == 'tied':
                assert num_units % num_heads == 0, \
                    'For `tied` projection, attention size/depth must be ' \
                    'divisible by the number of attention heads.'
                values_split = rops.split_heads(keys, num_heads)
            elif c.cnn_fm_projection == 'independent':
                assert num_units % num_heads == 0, \
                    'For `untied` projection, attention size/depth must be ' \
                    'divisible by the number of attention heads.'
                # Project and split memory
                v_layer = self._dense_layer(units=num_units, name='value_layer',
                                            use_bias=False, **self._mask_params)
                # (batch_size, num_heads, mem_size, num_units / num_heads)
                values_split = rops.split_heads(v_layer(values), num_heads)
            else:
                assert _shape(values)[-1] % num_heads == 0, \
                    'For `none` projection, feature map channel dim size must ' \
                    'be divisible by the number of attention heads.'
                values_split = rops.split_heads(values, num_heads)
            # query_keys = tf.concat([query, keys], axis=-1)

            # MLP
            v = tf.get_variable('attention_v', [num_units], dtype=self._dtype)
            if c.supermask_type:
                v, _ = masked_layer.generate_masks(
                    kernel=v, bias=None, dtype=self._dtype, **self._mask_params)
            att_score = query + keys
            att_score = tf.multiply(att_score, v)
            att_score = rops.split_heads(att_score, num_heads)  # (batch, heads, HW, att_size / heads)
            print('Attention multi head score shape: {}'.format(_shape(att_score)))
            att_score = tf.reduce_sum(att_score, axis=-1)
            assert len(_shape(att_score)) == 3  # (batch, heads, HW)

        # Compute attention context vector
        alignments = self.attn_maps = tf.nn.softmax(att_score, axis=-1)
        if len(_shape(alignments)) != 3:
            raise ValueError('Unexpected `alignments` shape: {}'.format(_shape(alignments)))
        alignments = self._maybe_dropout(alignments, c.attn_keep_prob)
        # Multi-head attention
        # Expand from [batch_size, num_heads, memory_time] to [batch_size, num_heads, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, 2)
        # attention_mechanism.values shape is
        #     [batch_size, num_heads, memory_time, num_units / num_heads]
        # the batched matmul is over memory_time, so the output shape is
        #     [batch_size, num_heads, 1, num_units / num_heads].
        # we then combine the heads
        #     [batch_size, 1, attention_mechanism.num_units]
        context = tf.matmul(expanded_alignments, values_split)
        attention = tf.squeeze(rops.combine_heads(context), [1])
        return attention

    def _attention_module_v2(self, query, feature_map):
        assert query.shape.ndims == 2, '`query` must be rank 2 with shape (batch, units)'
        assert feature_map.shape.ndims == 3, '`feature_map` must be rank 3 with shape (batch, memory_size, units)'
        c = self._config
        cnn_attention = self._get_attention_mech()(feature_map=feature_map, memory_sequence_length=None)
        alignments, _ = cnn_attention(query, state=None)
        # Compute attention context vector
        if len(_shape(alignments)) != 3:
            raise ValueError('Unexpected `alignments` shape: {}'.format(_shape(alignments)))
        alignments = self._maybe_dropout(alignments, c.attn_keep_prob)
        # Multi-head attention
        # Expand from [batch_size, num_heads, memory_time] to [batch_size, num_heads, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, 2)
        # attention_mechanism.values shape is
        #     [batch_size, num_heads, memory_time, num_units / num_heads]
        # the batched matmul is over memory_time, so the output shape is
        #     [batch_size, num_heads, 1, num_units / num_heads].
        # we then combine the heads
        #     [batch_size, 1, attention_mechanism.num_units]
        context = tf.matmul(expanded_alignments, cnn_attention.values_split)
        context = tf.squeeze(rops.combine_heads(context), [1])
        return context

    def _maybe_compute_classification_loss(self, use_sigmoid=True, sum_across_class=True):
        if not self.is_training:
            return None
        with tf.name_scope('loss'):
            if use_sigmoid:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.answers, logits=self.logits)
            else:
                answers = tf.nn.softmax(self.answers, axis=-1)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=answers, logits=self.logits)
            print('`loss` shape: {}'.format(_shape(loss)))
            if sum_across_class:
                loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            else:
                loss = tf.reduce_mean(loss)
            self.model_loss = self.snip_loss = loss
            tf.summary.scalar('loss', self.model_loss)
            eq = tf.equal(self.predictions, tf.argmax(self.answers, axis=-1))
            self.accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
            return self.model_loss
