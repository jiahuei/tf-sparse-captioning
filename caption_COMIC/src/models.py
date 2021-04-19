# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:12:32 2017

@author: jiahuei

"""
import logging
import tensorflow as tf
from src import model_base_v3 as base
from common import ops_v1 as ops

logger = logging.getLogger(__name__)
_shape = ops.shape


class CaptionModel(base.ModelBase):
    def __init__(self,
                 config,
                 mode,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        logger.debug('Building graph for: {}'.format(name))
        super().__init__(config, mode)
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._dtype = tf.float32
        
        # Start to build the model
        c = self._config
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                self._encoder()
            with tf.variable_scope('decoder'):
                self._decoder_rnn()
        
        # We place the optimisation graph out of 'Model' scope
        self._compute_caption_loss()
        self._maybe_add_train_op()
        
        # Maybe perform magnitude pruning
        self._maybe_add_magnitude_pruning_ops()
        
        if self.is_inference:
            if self.dec_attn_maps is None:
                self.infer_output = [self.dec_preds, tf.zeros([])]
            else:
                self.infer_output = [self.dec_preds, self.dec_attn_maps]
        else:
            # Log softmax temperature value
            t = tf.get_collection('softmax_temperatures')
            if len(t) > 0:
                tf.summary.scalar('softmax_temperature', t[0])
            self.summary_op = tf.summary.merge_all()
            logger.info('Model `{}` initialisation complete.'.format(mode))


class CaptionModelSCST(base.ModelBase):
    def __init__(self,
                 config,
                 scst_mode,
                 reuse=False):
        assert scst_mode in ['train', 'sample']
        # assert config.token_type == 'word'
        
        logger.debug('Building graph for: {}'.format(scst_mode))
        super().__init__(config, scst_mode if scst_mode == 'train' else 'infer')
        c = self._config
        batch_size = c.batch_size_train
        assert c.scst_beam_size >= 0
        if self.is_training:
            batch_size *= max(c.scst_beam_size, 1)
        im_size = c.cnn_input_size
        self.imgs = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, im_size[0], im_size[1], 3])
        self.captions = tf.placeholder_with_default(
            input=tf.zeros(shape=[batch_size, 1], dtype=tf.int32),
            shape=[batch_size, None])
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[batch_size])
        self.batch_ops = [self.imgs, self.captions]
        self.reuse = reuse
        self.name = scst_mode
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._dtype = tf.float32
        
        # Start to build the model
        vs_kwargs = dict(reuse=tf.AUTO_REUSE,
                         initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                self._encoder()
            with tf.variable_scope('decoder'):
                if self.is_training:
                    self._decoder_rnn_scst()
                else:
                    with tf.name_scope('greedy'):
                        self._decoder_rnn_scst(1)
                        self.dec_preds_greedy = self.dec_preds
                    with tf.name_scope('beam'):
                        self._decoder_rnn_scst(c.scst_beam_size)
                        self.dec_preds_beam = self.dec_preds
                    # with tf.name_scope('sample'):
                    #    self._decoder_rnn_scst(0)
                    #    self.dec_preds_sample = self.dec_preds
        
        # Generated captions can be obtained by calling self.dec_preds
        
        # We place the optimisation graph out of 'Model' scope
        self._compute_caption_loss(scst=True)
        self.train_scst = self._maybe_add_train_op()
        
        # Log softmax temperature value
        t = tf.get_collection('softmax_temperatures')
        if len(t) > 0: tf.summary.scalar('softmax_temperature', t[0])
        self.summary_op = tf.summary.merge_all()
        logger.info('Model `{}` initialisation complete.'.format(scst_mode))


class CaptionModelSparseTiming(base.ModelBase):
    def __init__(self,
                 config,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert config.is_sparse is True
        logger.debug('Building graph for: {}'.format(name))
        super().__init__(config, mode='infer')
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._dtype = tf.float32
        
        # Start to build the model
        c = self._config
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                self._encoder()
        
        # When in RNN timing mode, we store the CNN features and embedding in Variables
        # to try to exclude CNN execution latency
        with tf.variable_scope('activation_storage', **vs_kwargs):
            im_embed = self.im_embed
            cnn_fmaps = self.cnn_fmaps
            self.im_embed = tf.get_variable(name='im_embed',
                                            shape=_shape(im_embed),
                                            dtype=None)
            self.cnn_fmaps = tf.get_variable(name='cnn_fmaps',
                                             shape=_shape(cnn_fmaps),
                                             dtype=None)
            self.store_cnn_activations = [
                tf.assign(self.im_embed, value=im_embed),
                tf.assign(self.cnn_fmaps, value=cnn_fmaps)
            ]
        
        with tf.variable_scope('Model/decoder', **vs_kwargs):
            self._decoder_rnn()
        
        self.infer_output = self.dec_preds


class MNISTRNNModel(base.ModelBaseNoCNN):
    def __init__(self,
                 config,
                 mode,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        logger.debug('Building graph for: {}'.format(name))
        
        # Setting `itow` allows reuse of captioning code
        config.itow = list(range(10))
        super().__init__(config, mode)
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._dtype = tf.float32
        
        # Start to build the model
        c = self._config
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            with tf.variable_scope('encoder'):
                self._process_inputs()
                self._encoder_rnn()
                enc_out_layer = self._dense_layer(units=10, name='output_projection', **self._mask_params)
                enc_out_layer.build(c.rnn_size)
                self.dec_logits = tf.expand_dims(enc_out_layer(self.encoder_final_state), 1)  # (batch, 1, softmax)
        
        # We place the optimisation graph out of 'Model' scope
        # Reuse captioning loss as it is also softmax cross-entropy
        self._compute_caption_loss()
        self._maybe_add_train_op()
        
        # Maybe perform magnitude pruning
        self._maybe_add_magnitude_pruning_ops()
        
        if self.is_training:
            self.summary_op = tf.summary.merge_all()
            logger.info('Model `{}` initialisation complete.'.format(mode))
    
    def _process_inputs(self):
        """
        Generates the necessary inputs, targets, masks.
        """
        c = self._config
        images, decoder_inputs = self.batch_ops
        # Images are transposed from NHW (batch-major) to HNW (time-major) for row-by-row processing
        self.encoder_inputs = dict(input_seq=tf.transpose(images, [1, 0, 2]))
        self.decoder_inputs = dict(inputs=None,
                                   targets=decoder_inputs,  # [batch, 1]
                                   masks=tf.to_float(tf.ones_like(decoder_inputs)),
                                   seq_lens=None)


class VQAModel(base.VQABase):
    def __init__(self,
                 config,
                 mode,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        logger.debug('Building graph for: {}'.format(name))
        super().__init__(config, mode)
        c = self._config
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._softmax_size = len(c.answer_list)
        self._dtype = tf.float32
        
        # Start to build the model
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            # print('Embedding map shape: {}'.format(_shape(self._word_embed_map)))
            with tf.variable_scope('encoder'):
                _, cnn_feat_map = self._encoder()
                # print('CNN feature map shape: {}'.format(_shape(self.encoder_outputs)))
                # print('CNN final embed shape: {}'.format(_shape(self.encoder_final_state)))
            with tf.variable_scope('encoder/rnn'):
                embed_q, _ = self._encoder_rnn()
                # print('RNN output shape: {}'.format(_shape(self.encoder_outputs)))
                # print('RNN final embed shape: {}'.format(_shape(self.encoder_final_state)))
            with tf.variable_scope('attention'):
                # Normalise and concat with RNN final state
                cnn_feat_map = tf.nn.l2_normalize(cnn_feat_map, axis=-1)  # (N, H * W, C)
                attn_v = self._attention_module(
                    query=tf.expand_dims(embed_q, axis=1),  # (N, 1, rnn_size)
                    keys=cnn_feat_map, values=cnn_feat_map)
            with tf.variable_scope('multimodal_fusion'):
                embed_q = self._gated_tanh(embed_q, c.rnn_size, 'gated_tanh_q')
                embed_v = self._gated_tanh(attn_v, c.rnn_size, 'gated_tanh_v')
                embed_h = tf.multiply(embed_q, embed_v)
                print('`embed_q` shape: {}'.format(_shape(embed_q)))
                print('`embed_v` shape: {}'.format(_shape(embed_v)))
            with tf.variable_scope('classifier'):
                print('`embed_h` shape: {}'.format(_shape(embed_h)))
                embed_h = self._gated_tanh(embed_h, c.rnn_size)
                embed_h = self._maybe_dropout(embed_h, c.keep_prob)
                out_layer = self._dense_layer(
                    units=self._softmax_size, use_bias=False, name='projection', **self._mask_params)
                self.logits = out_layer(embed_h)  # (batch, softmax)
                self.predictions = tf.argmax(self.logits, axis=-1)
                print('`logits` shape: {}'.format(_shape(self.logits)))
        
        # We place the optimisation graph out of 'Model' scope
        self._maybe_compute_classification_loss()
        self._maybe_add_train_op()
        
        # Maybe perform magnitude pruning
        self._maybe_add_magnitude_pruning_ops()
        
        if self.is_inference:
            if self.attn_maps is None:
                self.infer_output = [self.predictions, tf.zeros([])]
            else:
                self.infer_output = [self.predictions, self.attn_maps]
        else:
            # Log softmax temperature value
            t = tf.get_collection('softmax_temperatures')
            if len(t) > 0:
                tf.summary.scalar('softmax_temperature', t[0])
            self.summary_op = tf.summary.merge_all()
            logger.info('Model `{}` initialisation complete.'.format(mode))


class VQAModelSimple(base.VQABase):
    """
    Based on this repo and commit:
    https://github.com/LeeDoYup/bottom-up-attention-tf/tree/0500c0717e9b9f65c8406bd3c2b30c8b83b36ebb
    
    More reference:
    https://github.com/hengyuan-hu/bottom-up-attention-vqa
    """
    
    def __init__(self,
                 config,
                 mode,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        logger.debug('Building graph for: {}'.format(name))
        super().__init__(config, mode)
        c = self._config
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._softmax_size = len(c.answer_list)
        self._dtype = tf.float32
        
        # Start to build the model
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            # print('Embedding map shape: {}'.format(_shape(self._word_embed_map)))
            if c.use_image_features is None:
                with tf.variable_scope('encoder'):
                    _, cnn_feat_map = self._encoder()
                    print('CNN feature map shape: {}'.format(_shape(self.encoder_outputs)))
                    print('CNN final embed shape: {}'.format(_shape(self.encoder_final_state)))
                # cnn_feat_map = tf.nn.l2_normalize(cnn_feat_map, axis=-1)  # (N, H * W, C)
            else:
                cnn_feat_map = self.batch_ops[0]
                print('Feature map shape: {}'.format(_shape(cnn_feat_map)))
            with tf.variable_scope('encoder/rnn'):
                embed_q, _ = self._encoder_rnn()
                print('RNN output shape: {}'.format(_shape(self.encoder_outputs)))
                print('RNN final embed shape: {}'.format(_shape(self.encoder_final_state)))
            with tf.variable_scope('attention'):
                # Normalise and concat with RNN final state
                attn_v = self._attention_module_v2(query=embed_q, feature_map=cnn_feat_map)
            with tf.variable_scope('multimodal_fusion'):
                embed_q = self._fully_connected(embed_q, c.rnn_size, 'embed_q', input_keep_prob=c.keep_prob)
                embed_v = self._fully_connected(attn_v, c.rnn_size, 'embed_v', input_keep_prob=c.keep_prob)
                embed_h = tf.multiply(embed_q, embed_v)
                # embed_h = tf.identity(embed_q)
                print('`embed_q` shape: {}'.format(_shape(embed_q)))
                print('`embed_v` shape: {}'.format(_shape(embed_v)))
            with tf.variable_scope('classifier'):
                print('`embed_h` shape: {}'.format(_shape(embed_h)))
                embed_h = self._fully_connected(
                    embed_h, c.rnn_size, 'h0', use_batch_norm=True, input_keep_prob=c.keep_prob)
                self.logits = self._fully_connected(
                    embed_h, self._softmax_size, 'projection',
                    activation_fn=None, input_keep_prob=c.keep_prob)
                self.predictions = tf.argmax(self.logits, axis=-1)
                print('`logits` shape: {}'.format(_shape(self.logits)))
        
        # We place the optimisation graph out of 'Model' scope
        self._maybe_compute_classification_loss(sum_across_class=False, use_sigmoid=True)
        self._maybe_add_train_op()
        
        # Maybe perform magnitude pruning
        self._maybe_add_magnitude_pruning_ops()
        
        if self.is_inference:
            if self.attn_maps is None:
                self.infer_output = [self.predictions, tf.zeros([])]
            else:
                self.infer_output = [self.predictions, self.attn_maps]
        else:
            # Log softmax temperature value
            t = tf.get_collection('softmax_temperatures')
            if len(t) > 0:
                tf.summary.scalar('softmax_temperature', t[0])
            self.summary_op = tf.summary.merge_all()
            logger.info('Model `{}` initialisation complete.'.format(mode))


class VQAModelLM(base.VQABase):
    """
    Based on this repo and commit:
    https://github.com/LeeDoYup/bottom-up-attention-tf/tree/0500c0717e9b9f65c8406bd3c2b30c8b83b36ebb

    More reference:
    https://github.com/hengyuan-hu/bottom-up-attention-vqa
    """
    
    def __init__(self,
                 config,
                 mode,
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        logger.debug('Building graph for: {}'.format(name))
        super().__init__(config, mode)
        c = self._config
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        self._softmax_size = len(c.answer_list)
        self._dtype = tf.float32
        
        # Start to build the model
        vs_kwargs = dict(reuse=tf.AUTO_REUSE, initializer=self._get_initialiser())
        
        if self.is_training:
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr()
            tf.summary.scalar('learning_rate', self.lr)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                _, cnn_feat_map = self._encoder()
            with tf.variable_scope('encoder/rnn'):
                embed_q, _ = self._encoder_rnn()
                print('RNN output shape: {}'.format(_shape(self.encoder_outputs)))
                print('RNN final embed shape: {}'.format(_shape(self.encoder_final_state)))
            with tf.variable_scope('classifier'):
                self.logits = self._fully_connected(
                    embed_q, self._softmax_size, 'projection',
                    activation_fn=None, input_keep_prob=c.keep_prob)
                self.predictions = tf.argmax(self.logits, axis=-1)
                print('`logits` shape: {}'.format(_shape(self.logits)))
        
        # We place the optimisation graph out of 'Model' scope
        self._maybe_compute_classification_loss(sum_across_class=False, use_sigmoid=True)
        self._maybe_add_train_op()
        self.summary_op = tf.summary.merge_all()
        logger.info('Model `{}` initialisation complete.'.format(mode))
