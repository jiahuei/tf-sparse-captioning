#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:22:42 2017

@author: jiahuei
"""
# import numpy as np
import logging
import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import _check_batch_beam, gather_tree_from_array
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense
# from tensorflow.python.layers import base
# from tensorflow.python.framework import dtypes
# from common.mask_prune import sampler
from common.mask_prune import masked_layer
from common.ops_v1 import layer_norm_activate, dprint
from common.ops_v1 import shape as _shape

# from packaging import version

AttentionWrapperState = tf.contrib.seq2seq.AttentionWrapperState

logger = logging.getLogger(__name__)
_DEBUG = False


def _dprint(string):
    return dprint(string, _DEBUG)


def _layer_norm_tanh(tensor):
    # if version.parse(tf.__version__) >= version.parse('1.9'):
    try:
        tensor = layer_norm_activate(
            'LN_tanh',
            tensor,
            tf.nn.tanh,
            begin_norm_axis=-1)
    except TypeError:
        tensor_s = _shape(tensor)
        tensor = layer_norm_activate(
            'LN_tanh',
            tf.reshape(tensor, [-1, tensor_s[-1]]),
            tf.nn.tanh)
        tensor = tf.reshape(tensor, tensor_s)
    return tensor


###############################################################################


def rnn_decoder_beam_search(cell,
                            embedding_fn,
                            output_layer,
                            batch_size,
                            beam_size,
                            length_penalty_weight,
                            maximum_iterations,
                            start_id,
                            end_id,
                            swap_memory=True):
    """
    Dynamic RNN loop function for inference. Performs beam search.
    Operates in time-major mode.

    Args:
        cell: An `RNNCell` instance (with or without attention).
        embedding_fn: Either embedding Variable or embedding function.
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        beam_size: `Int scalar. Size of beam for beam search.
        length_penalty_weight: Float weight to penalise length.
            Disabled with 0.0.
        maximum_iterations: Int scalar. Maximum number of decoding steps.
        start_id: `int32` scalar, the token that marks start of decoding.
        end_id: `int32` scalar, the token that marks end of decoding.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.

    Returns:
        top_sequence, top_score, None
    """
    logger.debug('Building subgraph V4 for Beam Search.')
    
    state_init = cell.zero_state(batch_size * beam_size, tf.float32)
    start_ids = tf.tile([start_id], multiples=[batch_size])
    _dprint('rnn_decoder_beam_search: Initial state: {}'.format(state_init))
    _dprint('rnn_decoder_beam_search: Cell state size: {}'.format(cell.state_size))
    
    # decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    decoder = BeamSearchDecoderMultiHead(
        cell=cell,
        embedding=embedding_fn,
        start_tokens=start_ids,
        end_token=end_id,
        initial_state=state_init,
        beam_width=beam_size,
        output_layer=output_layer,
        length_penalty_weight=length_penalty_weight,
        reorder_tensor_arrays=True)  # r1.9 API
    dec_outputs, dec_states, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations,
        parallel_iterations=1,
        swap_memory=swap_memory)
    _dprint('rnn_decoder_beam_search: Final BeamSearchDecoderState: {}'.format(dec_states))
    
    # `dec_outputs` will be a `FinalBeamSearchDecoderOutput` object
    # `dec_states` will be a `BeamSearchDecoderState` object
    predicted_ids = dec_outputs.predicted_ids  # (time, batch_size, beam_size)
    scores = dec_outputs.beam_search_decoder_output.scores  # (time, batch_size, beam_size)
    # top_sequence = predicted_ids[:, :, 0]
    # top_score = scores[:, :, 0]                                                 # log-softmax scores
    return predicted_ids, scores, dec_states.cell_state


def rnn_decoder_search(cell,
                       embedding_fn,
                       output_layer,
                       batch_size,
                       maximum_iterations,
                       start_id,
                       end_id,
                       swap_memory=True,
                       greedy_search=True):
    """
    Dynamic RNN loop function for inference. Performs greedy search / sampling.
    Operates in time-major mode.

    Args:
        cell: An `RNNCell` instance (with or without attention).
        embedding_fn: A callable that takes a vector tensor of `ids`
            (argmax ids), or the `params` argument for `embedding_lookup`.
            The returned tensor will be passed to the decoder input.
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        maximum_iterations: Int scalar. Maximum number of decoding steps.
        start_id: `int32` scalar, the token that marks start of decoding.
        end_id: `int32` scalar, the token that marks end of decoding.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.
        greedy_search: Python bool, use argmax if True, sample from
            distribution if False.

    Returns:
        output_ids, rnn_outputs, decoder_state
    """
    # Initialise `AttentionWrapperState` with provided RNN state
    state_init = cell.zero_state(batch_size, tf.float32)
    start_ids = tf.tile([start_id], multiples=[batch_size])
    _dprint('rnn_decoder_search: Initial state: {}'.format(state_init))
    _dprint('rnn_decoder_search: Cell state size: {}'.format(cell.state_size))
    
    if greedy_search:
        logger.debug('Building subgraph V4 for Greedy Search.')
        helper_fn = tf.contrib.seq2seq.GreedyEmbeddingHelper
    else:
        logger.debug('Building subgraph V4 for Sample Search.')
        helper_fn = tf.contrib.seq2seq.SampleEmbeddingHelper
    helper = helper_fn(
        embedding=embedding_fn,
        start_tokens=start_ids,
        end_token=end_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=state_init,
        output_layer=output_layer)
    dec_outputs, dec_states, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations,
        parallel_iterations=1,
        swap_memory=swap_memory)
    
    # `dec_outputs` will be a `BasicDecoderOutput` object
    # `dec_states` may be a `AttentionWrapperState` object
    rnn_out = dec_outputs.rnn_output
    output_ids = dec_outputs.sample_id
    
    return output_ids, rnn_out, dec_states


def rnn_decoder_training(cell,
                         embeddings,
                         output_layer,
                         batch_size,
                         sequence_length,
                         swap_memory=True):
    """
    Dynamic RNN loop function for training. Operates in time-major mode.
    The decoder will run until <EOS> token is encountered.

    Args:
        cell: An `RNNCell` instance (with or without attention).
        embeddings: A float32 tensor of shape [time, batch, word_size].
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        sequence_length: An int32 vector tensor. Length of sequence.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.

    Returns:
        output_ids, rnn_outputs, decoder_state
    """
    logger.debug('Building dynamic decode subgraph V4 for training.')
    
    # Initialise `AttentionWrapperState` with provided RNN state
    # batch_size = tf.shape(embeddings)[1]
    state_init = cell.zero_state(batch_size, tf.float32)
    _dprint('rnn_decoder_training: Initial state: {}'.format(state_init))
    _dprint('rnn_decoder_training: Cell state size: {}'.format(cell.state_size))
    
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=embeddings,
        sequence_length=sequence_length,
        time_major=True)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=state_init,
        output_layer=output_layer)
    dec_outputs, dec_states, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=True,
        impute_finished=True,
        maximum_iterations=None,
        parallel_iterations=1,
        swap_memory=swap_memory)
    
    # `dec_outputs` will be a `BasicDecoderOutput` object
    # `dec_states` may be a `AttentionWrapperState` object
    rnn_out = dec_outputs.rnn_output
    output_ids = dec_outputs.sample_id
    
    # Perform padding by copying elements from the last time step.
    # This is required if `impute_finished` is True.
    # This is skipped in inference mode.
    pad_time = tf.shape(embeddings)[0] - tf.shape(rnn_out)[0]
    pad = tf.tile(rnn_out[-1:, :, :], [pad_time, 1, 1])
    rnn_out = tf.concat([rnn_out, pad], axis=0)  # (max_time, batch_size, rnn_size)
    pad_ids = tf.tile(output_ids[-1:, :], [pad_time, 1])
    output_ids = tf.concat([output_ids, pad_ids], axis=0)  # (max_time, batch_size)
    
    return output_ids, rnn_out, dec_states


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer

    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    old_shape = _shape(x)
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [num_heads] + [last // num_heads if last else -1]
    # new_shape = tf.concat([old_shape[:-1], [num_heads, last // num_heads]], 0)
    return tf.transpose(tf.reshape(x, new_shape, 'split_head'), [0, 2, 1, 3])


def combine_heads(x):
    """Inverse of split_heads.

    Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

    Returns:
        a Tensor with shape [batch, length, channels]
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = _shape(x)
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else -1]
    # l = old_shape[2]
    # c = old_shape[3]
    # new_shape = tf.concat([old_shape[:-2] + [l * c]], 0)
    return tf.reshape(x, new_shape, 'combine_head')


###############################################################################


# noinspection PyProtectedMember
class MultiHeadAttV3(attention_wrapper._BaseAttentionMechanism):
    """
    Implements multi-head attention.
    """
    
    # TODO: bookmark
    # noinspection PyCallingNonCallable
    def __init__(self,
                 num_units,
                 feature_map,
                 fm_projection,
                 num_heads=None,
                 scale=True,
                 memory_sequence_length=None,
                 probability_fn=tf.nn.softmax,
                 mask_type=None,
                 mask_init_value=0,
                 mask_bern_sample=False,
                 name='MultiHeadAttV3'):
        """
        Construct the AttentionMechanism mechanism.
        Args:
            num_units: The depth of the attention mechanism.
            feature_map: The feature map / memory to query. This tensor
                should be shaped `[batch_size, height * width, channels]`.
            fm_projection: Feature map projection mode.
            num_heads: Int, number of attention heads. (optional)
            scale: Python boolean.  Whether to scale the energy term.
            memory_sequence_length: Tensor indicating sequence length.
            probability_fn: (optional) A `callable`.  Converts the score
                to probabilities.  The default is `tf.nn.softmax`.
            name: Name to use when creating ops.
        """
        logger.debug('Using MultiHeadAttV3.')
        assert fm_projection in [None, 'independent', 'tied']
        
        # if memory_sequence_length is not None:
        #     assert len(_shape(memory_sequence_length)) == 2, \
        #         '`memory_sequence_length` must be a rank-2 tensor, ' \
        #         'shaped [batch_size, num_heads].'
        
        if mask_type is None:
            self._dense_layer = Dense
            self._mask_params = {}
        else:
            self._dense_layer = masked_layer.MaskedDense
            self._mask_params = dict(mask_type=mask_type,
                                     mask_init_value=mask_init_value,
                                     mask_bern_sample=mask_bern_sample)
        
        super(MultiHeadAttV3, self).__init__(
            query_layer=self._dense_layer(units=num_units, name='query_layer', use_bias=False, **self._mask_params),
            # query is projected hidden state
            memory_layer=self._dense_layer(units=num_units, name='memory_layer', use_bias=False, **self._mask_params),
            # self._keys is projected feature_map
            memory=feature_map,  # self._values is feature_map
            probability_fn=lambda score, _: probability_fn(score),
            memory_sequence_length=None,
            score_mask_value=float('-inf'),
            name=name)
        
        self._probability_fn = lambda score, _: (
            probability_fn(
                self._maybe_mask_score_multi(
                    score, memory_sequence_length, float('-inf'))))
        self._fm_projection = fm_projection
        self._num_units = num_units
        self._num_heads = num_heads
        self._scale = scale
        self._feature_map_shape = _shape(feature_map)
        self._name = name
        
        if fm_projection == 'tied':
            assert num_units % num_heads == 0, \
                'For `tied` projection, attention size/depth must be ' \
                'divisible by the number of attention heads.'
            self._values_split = split_heads(self._keys, self._num_heads)
        elif fm_projection == 'independent':
            assert num_units % num_heads == 0, \
                'For `untied` projection, attention size/depth must be ' \
                'divisible by the number of attention heads.'
            # Project and split memory
            v_layer = self._dense_layer(units=num_units, name='value_layer', use_bias=False, **self._mask_params)
            # (batch_size, num_heads, mem_size, num_units / num_heads)
            self._values_split = split_heads(v_layer(self._values), self._num_heads)
        else:
            assert _shape(self._values)[-1] % num_heads == 0, \
                'For `none` projection, feature map channel dim size must ' \
                'be divisible by the number of attention heads.'
            self._values_split = split_heads(self._values, self._num_heads)
        
        _dprint('{}: FM projection type: {}'.format(
            self.__class__.__name__, fm_projection))
        _dprint('{}: Splitted values shape: {}'.format(
            self.__class__.__name__, _shape(self._values_split)))
        _dprint('{}: Values shape: {}'.format(
            self.__class__.__name__, _shape(self._values)))
        _dprint('{}: Keys shape: {}'.format(
            self.__class__.__name__, _shape(self._keys)))
        _dprint('{}: Feature map shape: {}'.format(
            self.__class__.__name__, _shape(feature_map)))
    
    @property
    def values_split(self):
        return self._values_split
    
    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.

        Returns:
            A `dtype` tensor shaped `[batch_size, alignments_size]`
            (`alignments_size` is the values' `max_time`).
        """
        del batch_size
        s = _shape(self.values_split)[:-1]
        init = tf.zeros(shape=[s[0], s[1] * s[2]], dtype=dtype)
        _dprint('{}: Initial alignments shape: {}'.format(self.__class__.__name__, _shape(init)))
        return init
    
    def _maybe_mask_score_multi(self,
                                score,
                                memory_sequence_length,
                                score_mask_value):
        if memory_sequence_length is None:
            return score
        message = 'All values in memory_sequence_length must greater than zero.'
        with tf.control_dependencies(
                [tf.assert_positive(memory_sequence_length, message=message)]):
            print(_shape(score))
            score_mask = tf.sequence_mask(
                memory_sequence_length, maxlen=tf.shape(score)[2])
            score_mask_values = score_mask_value * tf.ones_like(score)
            masked_score = tf.where(score_mask, score, score_mask_values)
            _dprint('{}: score shape: {}'.format(
                self.__class__.__name__, _shape(score)))
            _dprint('{}: masked_score shape: {}'.format(
                self.__class__.__name__, _shape(masked_score)))
            return masked_score


class MultiHeadAddLN(MultiHeadAttV3):
    """
    Implements Toronto-style (Xu et al.) attention scoring with layer norm,
    as described in:
    "Show, Attend and Tell: Neural Image Caption Generation with
    Visual Attention." ICML 2015. https://arxiv.org/abs/1502.03044
    """
    
    def __call__(self, query, state):
        """
        Score the query based on the keys and values.
        Args:
            query: RNN hidden state. Tensor of shape `[batch_size, num_units]`.
            state: IGNORED. Previous alignment values.
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        del state
        with tf.variable_scope(None, 'multi_add_attention', [query]):
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            proj_query = tf.expand_dims(self.query_layer(query), 1)
            v = tf.get_variable(
                'attention_v', [self._num_units], dtype=proj_query.dtype)
            if len(self._mask_params) > 0:
                v, _ = masked_layer.generate_masks(kernel=v, bias=None,
                                                   dtype=proj_query.dtype,
                                                   **self._mask_params)
            score = self._keys + proj_query
            score = _layer_norm_tanh(score)
            score = tf.multiply(score, v)
            score = split_heads(score, self._num_heads)  # (batch_size, num_heads, mem_size, num_units / num_heads)
            score = tf.reduce_sum(score, axis=3)  # (batch_size, num_heads, mem_size)
        
        if self._scale:
            softmax_temperature = tf.get_variable(
                'softmax_temperature',
                shape=[],
                dtype=tf.float32,
                initializer=tf.constant_initializer(5.0),
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             'softmax_temperatures'])
            score = tf.truediv(score, softmax_temperature)
        alignments = self._probability_fn(score, None)
        next_state = alignments
        _dprint('{}: Alignments shape: {}'.format(
            self.__class__.__name__, _shape(alignments)))
        return alignments, next_state


class MultiHeadAdd(MultiHeadAttV3):
    """
    Implements Toronto-style (Xu et al.) attention scoring,
    as described in:
    "Show, Attend and Tell: Neural Image Caption Generation with
    Visual Attention." ICML 2015. https://arxiv.org/abs/1502.03044
    """
    
    def __call__(self, query, state):
        """
        Score the query based on the keys and values.
        Args:
            query: RNN hidden state. Tensor of shape `[batch_size, num_units]`.
            state: IGNORED. Previous alignment values.
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        del state
        with tf.variable_scope(None, 'MultiHeadAdd', [query]):
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            proj_query = tf.expand_dims(self.query_layer(query), 1)
            v = tf.get_variable(
                'attention_v', [self._num_units], dtype=proj_query.dtype)
            if len(self._mask_params) > 0:
                v, _ = masked_layer.generate_masks(kernel=v,
                                                   bias=None,
                                                   dtype=proj_query.dtype,
                                                   **self._mask_params)
            score = self._keys + proj_query
            score = tf.nn.tanh(score)
            score = tf.multiply(score, v)
            score = split_heads(score, self._num_heads)  # (batch_size, num_heads, mem_size, num_units / num_heads)
            score = tf.reduce_sum(score, axis=3)  # (batch_size, num_heads, mem_size)
        alignments = self._probability_fn(score, None)
        next_state = alignments
        _dprint('{}: Alignments shape: {}'.format(
            self.__class__.__name__, _shape(alignments)))
        return alignments, next_state


class MultiHeadDot(MultiHeadAttV3):
    """
    Implements scaled dot-product scoring,
    as described in:
    "Attention is all you need." NIPS 2017.
    https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """
    
    def __call__(self, query, state):
        """
        Score the query based on the keys and values.
        Args:
            query: RNN hidden state. Tensor of shape `[batch_size, num_units]`.
            state: IGNORED. Previous alignment values.
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        del state
        with tf.variable_scope(None, 'MultiHeadDot', [query]):
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            proj_query = tf.expand_dims(self.query_layer(query), 1)  # (batch_size, 1, num_units)
            score = tf.multiply(self._keys, proj_query)
            score = split_heads(score, self._num_heads)  # (batch_size, num_heads, mem_size, num_units / num_heads)
            score = tf.reduce_sum(score, axis=3)  # (batch_size, num_heads, mem_size)
            score /= tf.sqrt(self._num_units / self._num_heads)
        alignments = self._probability_fn(score, None)
        next_state = alignments
        _dprint('{}: Alignments shape: {}'.format(
            self.__class__.__name__, _shape(alignments)))
        return alignments, next_state


# noinspection PyProtectedMember
class MultiHeadAttentionWrapperV3(attention_wrapper.AttentionWrapper):
    """
    Wraps another `RNNCell` with attention, similar to `AttentionWrapper`.
    Allows optional multi-head attention.

    Logits projection should be performed at the decoder by passing in
    an instance of `tf.layers.Layer`, as argument for `output_layer`.
    
    skip_att_threshold: If value is in range (0, 1), perform binarisation; else perform bernoulli sampling.
    """
    
    # TODO: bookmark
    def __init__(self,
                 context_layer=True,
                 alignments_keep_prob=1.0,
                 mask_type=None,
                 mask_init_value=0,
                 mask_bern_sample=False,
                 **kwargs):
        logger.debug('Using {}.'.format(self.__class__.__name__))
        super(MultiHeadAttentionWrapperV3, self).__init__(**kwargs)
        if len(self._attention_mechanisms) != 1:
            raise ValueError('Only a single attention mechanism can be used.')
        
        self._context_layer = context_layer
        self._alignments_keep_prob = alignments_keep_prob
        
        if mask_type is None:
            self._dense_layer = Dense
            self._mask_params = {}
        else:
            self._dense_layer = masked_layer.MaskedDense
            self._mask_params = dict(mask_type=mask_type,
                                     mask_init_value=mask_init_value,
                                     mask_bern_sample=mask_bern_sample)
    
    # noinspection PyCallingNonCallable
    def call(self, inputs, prev_state):
        """
        Perform a step of attention-wrapped RNN.

        This method assumes `inputs` is the word embedding vector.

        This method overrides the original `call()` method.
        """
        _attn_mech = self._attention_mechanisms[0]
        attn_size = _attn_mech._num_units
        batch_size = _attn_mech.batch_size
        dtype = inputs.dtype
        
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        # `_cell_input_fn` defaults to
        # `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`
        _dprint('{}: prev_state received by call(): {}'.format(
            self.__class__.__name__, prev_state))
        cell_inputs = self._cell_input_fn(inputs, prev_state.attention)
        prev_cell_state = prev_state.cell_state
        cell_output, curr_cell_state = self._cell(cell_inputs, prev_cell_state)
        
        cell_batch_size = (cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory (encoder output) "
                "and the query (decoder output). Are you using the "
                "BeamSearchDecoder? You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                [tf.assert_equal(cell_batch_size, _attn_mech.batch_size, message=error_message)]):
            cell_output = tf.identity(cell_output, name="checked_cell_output")
        
        dtype = cell_output.dtype
        assert len(self._attention_mechanisms) == 1
        _attn_mech = self._attention_mechanisms[0]
        alignments, attention_state = _attn_mech(cell_output, state=None)
        
        if self._alignments_keep_prob < 1.:
            alignments = tf.contrib.layers.dropout(inputs=alignments,
                                                   keep_prob=self._alignments_keep_prob,
                                                   noise_shape=None,
                                                   is_training=True)
        
        if len(_shape(alignments)) == 3:
            # Multi-head attention
            # Expand from [batch_size, num_heads, memory_time] to [batch_size, num_heads, 1, memory_time]
            expanded_alignments = tf.expand_dims(alignments, 2)
            # attention_mechanism.values shape is
            #     [batch_size, num_heads, memory_time, num_units / num_heads]
            # the batched matmul is over memory_time, so the output shape is
            #     [batch_size, num_heads, 1, num_units / num_heads].
            # we then combine the heads
            #     [batch_size, 1, attention_mechanism.num_units]
            attention_mechanism_values = _attn_mech.values_split
            context = tf.matmul(expanded_alignments, attention_mechanism_values)
            attention = tf.squeeze(combine_heads(context), [1])
        else:
            # Expand from [batch_size, memory_time] to [batch_size, 1, memory_time]
            expanded_alignments = tf.expand_dims(alignments, 1)
            # Context is the inner product of alignments and values along the
            # memory time dimension.
            # alignments shape is
            #     [batch_size, 1, memory_time]
            # attention_mechanism.values shape is
            #     [batch_size, memory_time, attention_mechanism.num_units]
            # the batched matmul is over memory_time, so the output shape is
            #     [batch_size, 1, attention_mechanism.num_units].
            # we then squeeze out the singleton dim.
            attention_mechanism_values = _attn_mech.values
            context = tf.matmul(expanded_alignments, attention_mechanism_values)
            attention = tf.squeeze(context, [1])
        
        # Context projection
        if self._context_layer:
            # noinspection PyCallingNonCallable
            attention = self._dense_layer(name='a_layer',
                                          units=_attn_mech._num_units,
                                          use_bias=False,
                                          activation=None,
                                          dtype=dtype,
                                          **self._mask_params)(attention)
        
        if self._alignment_history:
            alignments = tf.reshape(alignments, [cell_batch_size, -1])
            alignment_history = prev_state.alignment_history.write(prev_state.time, alignments)
        else:
            alignment_history = ()
        
        curr_state = attention_wrapper.AttentionWrapperState(
            time=prev_state.time + 1,
            cell_state=curr_cell_state,
            attention=attention,
            attention_state=alignments,
            alignments=alignments,
            alignment_history=alignment_history
        )
        return cell_output, curr_state
    
    @property
    def state_size(self):
        state = super(MultiHeadAttentionWrapperV3, self).state_size
        _attn_mech = self._attention_mechanisms[0]
        s = _shape(_attn_mech._values_split)[1:3]
        state = state._replace(alignments=s[0] * s[1],
                               alignment_history=s[0] * s[1] if self._alignment_history else (),
                               attention_state=s[0] * s[1])
        if _attn_mech._fm_projection is None and self._context_layer is False:
            state = state.clone(attention=_attn_mech._feature_map_shape[-1])
        else:
            state = state.clone(attention=_attn_mech._num_units)
        _dprint('{}: state_size: {}'.format(self.__class__.__name__, state))
        return state
    
    # noinspection PyProtectedMember
    def zero_state(self, batch_size, dtype):
        state = super(MultiHeadAttentionWrapperV3, self).zero_state(
            batch_size, dtype)
        _attn_mech = self._attention_mechanisms[0]
        tf_ary_kwargs = dict(dtype=dtype,
                             size=0,
                             dynamic_size=True,
                             element_shape=None)
        if _attn_mech._fm_projection is None and self._context_layer is False:
            state = state._replace(
                attention=tf.zeros(
                    [batch_size, _attn_mech._feature_map_shape[-1]], dtype),
                alignment_history=tf.TensorArray(**tf_ary_kwargs) if self._alignment_history else ())
        else:
            state = state._replace(
                attention=tf.zeros(
                    [batch_size, _attn_mech._num_units], dtype),
                alignment_history=tf.TensorArray(**tf_ary_kwargs) if self._alignment_history else ())
        _dprint('{}: zero_state: {}'.format(self.__class__.__name__, state))
        return state


class BeamSearchDecoderMultiHead(tf.contrib.seq2seq.BeamSearchDecoder):
    # noinspection PyProtectedMember
    def _maybe_sort_array_beams(self, t, parent_ids, sequence_length):
        """Maybe sorts beams within a `TensorArray`.

        Args:
          t: A `TensorArray` of size `max_time` that contains `Tensor`s of shape
            `[batch_size, beam_width, s]` or `[batch_size * beam_width, s]` where
            `s` is the depth shape.
          parent_ids: The parent ids of shape `[max_time, batch_size, beam_width]`.
          sequence_length: The sequence length of shape `[batch_size, beam_width]`.

        Returns:
          A `TensorArray` where beams are sorted in each `Tensor` or `t` itself if
          it is not a `TensorArray` or does not meet shape requirements.
        """
        if not isinstance(t, tf.TensorArray):
            return t
        # pylint: disable=protected-access
        if (not t._infer_shape or not t._element_shape
                or t._element_shape[0].ndims is None
                or t._element_shape[0].ndims < 1):
            shape = (
                t._element_shape[0] if t._infer_shape and t._element_shape
                else tf.TensorShape(None))
            tf.logger.warn("The TensorArray %s in the cell state is not amenable to "
                            "sorting based on the beam search result. For a "
                            "TensorArray to be sorted, its elements shape must be "
                            "defined and have at least a rank of 1, but saw shape: %s"
                            % (t.handle.name, shape))
            return t
        # shape = t._element_shape[0]
        # pylint: enable=protected-access
        # if not _check_static_batch_beam_maybe(
        #    shape, tensor_util.constant_value(self._batch_size), self._beam_width):
        #    return t
        t = t.stack()
        with tf.control_dependencies(
                [_check_batch_beam(t, self._batch_size, self._beam_width)]):
            return gather_tree_from_array(t, parent_ids, sequence_length)
