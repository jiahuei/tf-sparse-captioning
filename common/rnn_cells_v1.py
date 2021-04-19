# -*- coding: utf-8 -*-
"""
Created on 17 Jul 2019 00:12:51

@author: jiahuei
"""
import logging
import tensorflow as tf
from common.mask_prune import masked_layer

logger = logging.getLogger(__name__)


def get_rnn_cell(name,
                 num_units,
                 reuse,
                 use_fused_cell=False,
                 use_masked_cell=False,
                 use_sparse_cell=False,
                 masked_cell_kwargs=None):
    if use_masked_cell:
        assert masked_cell_kwargs is not None
        assert len(masked_cell_kwargs) > 0
        if not use_fused_cell:
            logger.warning('Masked cells always use fused variants.')
        if use_sparse_cell:
            logger.warning('Masked cells does not have sparse variants.')

        if name == 'LSTM':
            return masked_layer.MaskedBasicLSTMCell(num_units=num_units,
                                                    reuse=reuse,
                                                    **masked_cell_kwargs)
        elif name == 'LN_LSTM':
            raise ValueError('LayerNormLSTM is not implemented for Supermasks.')
        elif name == 'GRU':
            return masked_layer.MaskedGRUCell(num_units=num_units,
                                              reuse=reuse,
                                              **masked_cell_kwargs)
        else:
            raise ValueError('Invalid RNN choice.')

    elif use_sparse_cell:
        assert not use_masked_cell
        if use_fused_cell:
            logger.warning('NOTE: Sparse cells does not have fused variants.')
        pass

    else:
        assert not use_masked_cell
        if name == 'LSTM':
            if use_fused_cell:
                return tf.contrib.rnn.LSTMBlockCell(num_units=num_units,
                                                    forget_bias=1.0,
                                                    cell_clip=-1,
                                                    use_peephole=False,
                                                    reuse=reuse,
                                                    name='basic_lstm_cell')
            else:
                return tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                                    state_is_tuple=True,
                                                    reuse=reuse)
        elif name == 'LN_LSTM':
            if use_fused_cell:
                logger.warning('`LN_LSTM` cells does not have fused variants.')
            return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units, reuse=reuse)

        elif name == 'GRU':
            if use_fused_cell:
                return tf.contrib.rnn.GRUBlockCellV2(num_units=num_units, reuse=reuse, name='gru_cell')
            else:
                return tf.contrib.rnn.GRUCell(num_units=num_units, reuse=reuse)

        # elif name == 'R_GRU':
        #     return ResidualGRUCell(num_units=num_units, reuse=reuse)
        # elif name == 'RRN':
        #     return ResidualRNNCell(num_units=num_units, reuse=reuse)
        # elif name == 'LN_RRN':
        #     return ResidualLayerNormRNNCell(num_units=num_units, reuse=reuse)
        # elif name == 'LN_RRN_B':
        #     return ResidualLayerNormRNNCellB(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_a1':
        #     return ResidualPreLayerNormRNNCellA1(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_a2':
        #     return ResidualPreLayerNormRNNCellA2(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_b1':
        #     return ResidualPreLayerNormRNNCellB1(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_b2':
        #     return ResidualPreLayerNormRNNCellB2(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_c1':
        #     return ResidualPreLayerNormRNNCellC1(num_units=num_units, reuse=reuse)
        # elif name == 'preLN_RRN_c2':
        #     return ResidualPreLayerNormRNNCellC2(num_units=num_units, reuse=reuse)
        else:
            raise ValueError('Invalid RNN choice.')


class ResidualGRUCell(tf.contrib.rnn.GRUCell):
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        gate_inputs = tf.matmul(tf.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

        value = tf.nn.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = tf.matmul(tf.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = tf.nn.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        res_h = u * state + (1 - u) * c
        new_h = tf.add(state, res_h)
        return new_h, new_h


class ResidualRNNCell(tf.contrib.rnn.BasicRNNCell):
    def call(self, inputs, state):
        """
        Residual Recurrent Network: output = new_state = old_state + act(W * input + U * state + B)
        """
        res_state = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
        res_state = tf.nn.bias_add(res_state, self._bias)
        res_state = self._activation(res_state)
        output = tf.add(res_state, state)
        return output, output


class ResidualLayerNormRNNCell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 norm_gain=1.0,
                 norm_shift=0.0):
        super(ResidualLayerNormRNNCell, self).__init__(
            num_units=num_units,
            activation=activation,
            reuse=reuse,
            name=name,
            dtype=dtype)
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        kernel_s = [input_depth + self._num_units, self._num_units]
        self._kernel = self.add_variable("kernel", shape=kernel_s)
        self.built = True

    # Residual Recurrent Network with Layer Norm
    def _norm(self, inp, scope, dtype=tf.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = tf.constant_initializer(self._norm_gain)
        beta_init = tf.constant_initializer(self._norm_shift)
        with tf.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            tf.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            tf.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def call(self, inputs, state):
        """
        Residual Recurrent Network with Layer Normalisation:
        output = new_state = old_state + act( LN( W * input + U * state ) )
        """
        res_state = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
        res_state = self._norm(res_state, 'state_LN', dtype=inputs.dtype)
        res_state = self._activation(res_state)
        output = tf.add(res_state, state)
        return output, output


class ResidualLayerNormRNNCellB(ResidualLayerNormRNNCell):
    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        kernel_s = [input_depth + self._num_units, 2 * self._num_units]
        self._kernel = self.add_variable("kernel", shape=kernel_s)
        self.built = True

    def call(self, inputs, state):
        """
        Residual Recurrent Network with Layer Normalisation:
        output = new_state = old_state + act( LN( W * input + U * state ) )
        """
        res_state = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
        res_state = self._norm(res_state, 'state_LN', dtype=inputs.dtype)
        res_state = tf.split(value=res_state, num_or_size_splits=2, axis=1)
        res_state = tf.multiply(tf.nn.sigmoid(res_state[0]), self._activation(res_state[1]))
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellA1(ResidualLayerNormRNNCell):
    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type A1):
        output = new_state = old_state + act( W * LN( [input, state] ) )
        """
        res_state = self._norm(tf.concat([inputs, state], 1), 'pre_LN', dtype=inputs.dtype)
        res_state = tf.matmul(res_state, self._kernel)
        res_state = self._activation(res_state)
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellA2(ResidualLayerNormRNNCell):
    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type A2):
        output = new_state = old_state + act( W * input + U * LN( state ) )
        """
        res_state = tf.concat([inputs, self._norm(state, 'pre_LN', dtype=inputs.dtype)], 1)
        res_state = tf.matmul(res_state, self._kernel)
        res_state = self._activation(res_state)
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellB1(ResidualLayerNormRNNCell):
    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type B1):
        output = new_state = old_state + W * act( LN( [input, state] ) )
        """
        res_state = self._norm(tf.concat([inputs, state], 1), 'pre_LN', dtype=inputs.dtype)
        res_state = self._activation(res_state)
        res_state = tf.matmul(res_state, self._kernel)
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellB2(ResidualLayerNormRNNCell):
    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type B2):
        output = new_state = old_state + W * act( [input, LN( state )] )
        """
        res_state = tf.concat([inputs, self._norm(state, 'pre_LN', dtype=inputs.dtype)], 1)
        res_state = self._activation(res_state)
        res_state = tf.matmul(res_state, self._kernel)
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellC1(ResidualLayerNormRNNCell):
    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        kernel_s = [input_depth + self._num_units, 2 * self._num_units]
        self._kernel = self.add_variable("kernel", shape=kernel_s)
        self.built = True

    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type C1):
        output = new_state = old_state + act( W * input + U * LN( state ) )
        """
        res_state = self._norm(tf.concat([inputs, state], 1), 'pre_LN', dtype=inputs.dtype)
        res_state = tf.matmul(res_state, self._kernel)
        res_state = tf.split(value=res_state, num_or_size_splits=2, axis=1)
        res_state = tf.multiply(tf.nn.sigmoid(res_state[0]), self._activation(res_state[1]))
        output = tf.add(res_state, state)
        return output, output


class ResidualPreLayerNormRNNCellC2(ResidualLayerNormRNNCell):
    # noinspection PyAttributeOutsideInit
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        kernel_s = [input_depth + self._num_units, 2 * self._num_units]
        self._kernel = self.add_variable("kernel", shape=kernel_s)
        self.built = True

    def call(self, inputs, state):
        """
        Residual Recurrent Network with pre Layer Normalisation (type C2):
        output = new_state = old_state + act( W * input + U * LN( state ) )
        """
        res_state = tf.concat([inputs, self._norm(state, 'pre_LN', dtype=inputs.dtype)], 1)
        res_state = tf.matmul(res_state, self._kernel)
        res_state = tf.split(value=res_state, num_or_size_splits=2, axis=1)
        res_state = tf.multiply(tf.nn.sigmoid(res_state[0]), self._activation(res_state[1]))
        output = tf.add(res_state, state)
        return output, output
