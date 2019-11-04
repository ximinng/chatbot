# encoding: utf-8

"""
    SequenceToSequence模型
    定义了模型编码器、解码器、优化器、训练、预测
"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, LSTMStateTuple, DropoutWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, TrainingHelper, BasicDecoder, \
    BeamSearchDecoder
from tensorflow import layers
from DataProcessing import DataUnit
from tensorflow.python.ops import array_ops


class Seq2Seq(object):

    def __init__(self, hidden_size, cell_type,
                 layer_size, batch_size,
                 encoder_vocab_size, decoder_vocab_size,
                 embedding_dim, share_embedding,
                 max_decode_step, max_gradient_norm,
                 learning_rate, decay_step,
                 min_learning_rate, bidirection,
                 beam_width,
                 mode
                 ):
        """
        初始化函数
        """
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.share_embedding = share_embedding
        self.max_decode_step = max_decode_step
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.min_learning_rate = min_learning_rate
        self.bidirection = bidirection
        self.beam_width = beam_width
        self.mode = mode
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.build_model()

    def build_model(self):
        """
        构建完整的模型
        :return:
        """
        self.init_placeholder()
        self.embedding()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)
        if self.mode == 'train':
            self.build_optimizer()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        """
        定义各个place_holder
        :return:
        """
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='encoder_inputs_length')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size, ],
                                                        name='decoder_inputs_length')
            self.decoder_start_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * DataUnit.START_INDEX
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=1)

    def embedding(self):
        """
        词嵌入操作
        :param share:编码器和解码器是否共用embedding
        :return:
        """
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(
                tf.truncated_normal(shape=[self.encoder_vocab_size, self.embedding_dim], stddev=0.1),
                name='encoder_embeddings')
            if not self.share_embedding:
                decoder_embedding = tf.Variable(
                    tf.truncated_normal(shape=[self.decoder_vocab_size, self.embedding_dim], stddev=0.1),
                    name='decoder_embeddings')
                self.encoder_embeddings = encoder_embedding
                self.decoder_embeddings = decoder_embedding
            else:
                self.encoder_embeddings = encoder_embedding
                self.decoder_embeddings = encoder_embedding

    def one_cell(self, hidden_size, cell_type):
        """
        一个神经元
        :return:
        """
        if cell_type == 'gru':
            c = GRUCell
        else:
            c = LSTMCell
        cell = c(hidden_size)
        cell = DropoutWrapper(
            cell,
            dtype=tf.float32,
            output_keep_prob=self.keep_prob,
        )
        cell = ResidualWrapper(cell)
        return cell

    def build_encoder_cell(self, hidden_size, cell_type, layer_size):
        """
        构建编码器所有层
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        cells = [self.one_cell(hidden_size, cell_type) for _ in range(layer_size)]
        return MultiRNNCell(cells)

    def build_encoder(self):
        """
        构建完整编码器
        :return:
        """
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
            encoder_inputs_embedded = layers.dense(encoder_inputs_embedded,
                                                   self.hidden_size,
                                                   use_bias=False,
                                                   name='encoder_residual_projection')
            initial_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)
            if self.bidirection:
                encoder_cell_bw = self.build_encoder_cell(self.hidden_size, self.cell_type, self.layer_size)
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=encoder_cell_bw,
                    cell_fw=encoder_cell,
                    inputs=encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    swap_memory=True)

                encoder_outputs = tf.concat(
                    (encoder_bw_outputs, encoder_fw_outputs), 2)
                encoder_final_state = []
                for i in range(self.layer_size):
                    c_fw, h_fw = encoder_fw_state[i]
                    c_bw, h_bw = encoder_bw_state[i]
                    c = tf.concat((c_fw, c_bw), axis=-1)
                    h = tf.concat((h_fw, h_bw), axis=-1)
                    encoder_final_state.append(LSTMStateTuple(c=c, h=h))
                encoder_final_state = tuple(encoder_final_state)
            else:
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    initial_state=initial_state,
                    swap_memory=True)

            return encoder_outputs, encoder_final_state

    def build_decoder_cell(self, encoder_outputs, encoder_final_state,
                           hidden_size, cell_type, layer_size):
        """
        构建解码器所有层
        :param encoder_outputs:
        :param encoder_state:
        :param hidden_size:
        :param cell_type:
        :param layer_size:
        :return:
        """
        sequence_length = self.encoder_inputs_length
        if self.mode == 'decode':
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=self.beam_width)
            sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=self.beam_width)

        if self.bidirection:
            cell = MultiRNNCell([self.one_cell(hidden_size * 2, cell_type) for _ in range(layer_size)])
        else:
            cell = MultiRNNCell([self.one_cell(hidden_size, cell_type) for _ in range(layer_size)])
        # 使用attention机制
        self.attention_mechanism = BahdanauAttention(
            num_units=self.hidden_size,
            memory=encoder_outputs,
            memory_sequence_length=sequence_length
        )

        def cell_input_fn(inputs, attention):
            mul = 2 if self.bidirection else 1
            attn_projection = layers.Dense(self.hidden_size * mul,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_size,
            cell_input_fn=cell_input_fn,
            name='Attention_Wrapper'
        )
        if self.mode == 'decode':
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size * self.beam_width,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size=self.batch_size,
                                                    dtype=tf.float32).clone(
                cell_state=encoder_final_state)
        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_final_state):
        """
        构建完整解码器
        :return:
        """
        with tf.variable_scope("decode"):
            decoder_cell, decoder_initial_state = self.build_decoder_cell(encoder_outputs, encoder_final_state,
                                                                          self.hidden_size, self.cell_type,
                                                                          self.layer_size)
            # 输出层投影
            decoder_output_projection = layers.Dense(self.decoder_vocab_size, dtype=tf.float32,
                                                     use_bias=False,
                                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                        stddev=0.1),
                                                     name='decoder_output_projection')
            if self.mode == 'train':
                # 训练模式
                decoder_inputs_embdedded = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs_train)
                training_helper = TrainingHelper(
                    inputs=decoder_inputs_embdedded,
                    sequence_length=self.decoder_inputs_length,
                    name='training_helper'
                )
                training_decoder = BasicDecoder(decoder_cell, training_helper,
                                                decoder_initial_state, decoder_output_projection)
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length)
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  maximum_iterations=max_decoder_length)
                self.masks = tf.sequence_mask(self.decoder_inputs_length, maxlen=max_decoder_length, dtype=tf.float32,
                                              name='masks')
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=training_decoder_output.rnn_output,
                                                             targets=self.decoder_inputs,
                                                             weights=self.masks,
                                                             average_across_timesteps=True,
                                                             average_across_batch=True
                                                             )
            else:
                # 预测模式
                start_token = [DataUnit.START_INDEX] * self.batch_size
                end_token = DataUnit.END_INDEX
                inference_decoder = BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=lambda x: tf.nn.embedding_lookup(self.decoder_embeddings, x),
                    start_tokens=start_token,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=decoder_output_projection
                )
                inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                   maximum_iterations=self.max_decode_step)
                self.decoder_pred_decode = inference_decoder_output.predicted_ids
                self.decoder_pred_decode = tf.transpose(
                    self.decoder_pred_decode,
                    perm=[0, 2, 1]
                )

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, keep_prob, decode):
        """
            检查输入,返回输入字典
        """
        input_batch_size = encoder_inputs.shape[0]
        assert input_batch_size == encoder_inputs_length.shape[0], 'encoder_inputs 和 encoder_inputs_length的第一个维度必须一致'
        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            assert target_batch_size == input_batch_size, 'encoder_inputs 和 decoder_inputs的第一个维度必须一致'
            assert target_batch_size == decoder_inputs_length.shape[
                0], 'decoder_inputs 和 decoder_inputs_length的第一个维度必须一致'
        input_feed = {self.encoder_inputs.name: encoder_inputs,
                      self.encoder_inputs_length.name: encoder_inputs_length}
        input_feed[self.keep_prob.name] = keep_prob
        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build_optimizer(self):
        """
        构建优化器
        :return:
        """
        learning_rate = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                                  self.decay_step, self.min_learning_rate, power=0.5)
        self.current_learning_rate = learning_rate
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 优化器
        self.opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )
        # 梯度裁剪
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm
        )
        # 更新梯度
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step
        )

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, keep_prob):
        """
        训练模型
        :param sess:
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, keep_prob,
                                      False)
        output_feed = [
            self.update, self.loss,
            self.current_learning_rate
        ]
        _, cost, lr = sess.run(output_feed, input_feed)
        return cost, lr

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        """
        预测
        :return:
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      None, None, 1, True)
        pred = sess.run(self.decoder_pred_decode, input_feed)
        return pred[0]

    def save(self, sess, save_path='model/chatbot_model.ckpt'):
        """
        保存模型
        :return:
        """
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='model/chatbot_model.ckpt'):
        """
        加载模型
        """
        self.saver.restore(sess, save_path)
