from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
LSTMCell      = tf.nn.rnn_cell.LSTMCell


class HierarchicalAttentionAutoencoder(object):
    def __init__(self, vocab, max_sent_len, max_doc_len, size=128, batch_size=1,
                 num_samples=512, checkpoint_dir="checkpoint"):
        #self.sess       = sess
        self.size           = size
        self.batch_size     = batch_size
        self.vocab          = vocab
        self.vocab_size     = len(vocab)
        self.sent_steps     = max_sent_len
        self.doc_steps      = max_doc_len
        self.num_samples    = num_samples
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        #max_sent_len, max_doc_len = create_vocabulary(self.vocab_file,
        #                                              self.data_file,
        #                                              self.vocab_size)
        #vocab, rev_vocab = initialize_vocabulary(self.vocab_file)
        #self.rev_vocab  = rev_vocab
        #self.vocab      = vocab
        #self.vocab_size = len(vocab)
        #self.sent_steps = max_sent_len
        #self.doc_steps  = max_doc_len

        self.input_data  = tf.placeholder(tf.int32, [self.batch_size,
                                                     self.doc_steps,
                                                     self.sent_steps])
        self.output_data = tf.placeholder(tf.int32, [self.batch_size,
                                                     self.doc_steps,
                                                     self.sent_steps])
        with tf.device("/gpu:0"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.size])
            self.inputs = []
            for s in range(self.doc_steps):
                self.inputs.append(tf.nn.embedding_lookup(self.embedding,
                                                          self.input_data[:,s,:]))
        # ENCODE
        # TODO: make this multi-layer
        self.encode_word_cell = BasicLSTMCell(self.size)
        self.encode_sent_cell = BasicLSTMCell(self.size)

        word_state   = self.encode_word_cell.zero_state(self.batch_size, tf.float32)
        sent_state   = self.encode_sent_cell.zero_state(self.batch_size, tf.float32)
        sent_encodes = []
        with tf.variable_scope("encode_RNN_word"):
            for s in range(self.doc_steps):
                for t in range(self.sent_steps):
                    if t > 0 or s > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, word_state) = self.encode_word_cell(self.inputs[s][:,t,:],
                                                                      word_state)
                sent_encode = cell_output
                sent_encodes.append(sent_encode)

        with tf.variable_scope("encode_RNN_sentence"):
            for s in range(self.doc_steps):
                if s > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, sent_state) = self.encode_sent_cell(sent_encodes[s],
                                                                  sent_state)
            doc_encode = cell_output

        self.doc_encode = doc_encode

        # DECODE
        self.decode_sent_cell = LSTMCell(2*self.size, num_proj=self.size)
        self.decode_word_cell = BasicLSTMCell(self.size)

        sent_decode  = tf.zeros((self.batch_size, self.size))
        sent_state   = self._build_zero_state(doc_encode)
        self.logits  = []
        sent_decodes = [sent_decode]
        word_decodes = []

        for s1 in range(self.doc_steps):
            vs = []
            for s2 in range(self.doc_steps):
                with tf.variable_scope("decode_RNN_attention"):
                    if s1 > 0 or s2 > 0: tf.get_variable_scope().reuse_variables()
                    u  = tf.get_variable("u", [self.size, 1])
                    w1 = tf.get_variable("w1", [self.size, self.size])
                    w2 = tf.get_variable("w2", [self.size, self.size])
                    b  = tf.get_variable("b1", [self.size])

                    vi = tf.matmul(tf.tanh(tf.add(tf.add(
                        tf.matmul(sent_decodes[s1], w1),
                        tf.matmul(sent_encodes[s2], w2)), b)), u)
                    vi = tf.exp(vi)
                    vs.append(vi)

            vs_sum = tf.add_n(vs)
            mt = tf.add_n([vs[i]*sent_encodes[i] for i in range(self.doc_steps)])/vs_sum

            with tf.variable_scope("decode_RNN_sentence"):
                if s1 > 0: tf.get_variable_scope().reuse_variables()
                sent_decode = tf.concat(1, [sent_decode, mt])
                (cell_output, sent_state) = self.decode_sent_cell(sent_decode,
                                                                  sent_state)
                sent_decode = cell_output
                sent_decodes.append(sent_decode)

                word_decode = tf.zeros((self.batch_size, self.size))
                word_state  = self._build_zero_state(sent_decode)
                with tf.variable_scope("decode_RNN_word"):
                    for t in range(self.sent_steps):
                        if t > 0 or s1 > 0: tf.get_variable_scope().reuse_variables()
                        (cell_output, word_state) = self.decode_word_cell(word_decode,
                                                                          word_state)
                        word_decode = cell_output
                        word_decodes.append(word_decode)

                    sent_state = word_state

        w = tf.get_variable("w", [self.size, self.vocab_size])
        b = tf.get_variable("b", [self.vocab_size])

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs,
                                              labels, self.num_samples,
                                              self.vocab_size)

        targets = tf.reshape(self.output_data, [self.batch_size,
                                                self.doc_steps*self.sent_steps])
        targets = [targets[:,i] for i in range(self.doc_steps*self.sent_steps)]
        weights = [tf.ones([self.batch_size]) for _ in range(self.doc_steps*self.sent_steps)]
        loss    = tf.nn.seq2seq.sequence_loss_by_example(word_decodes, targets, weights,
                                                         softmax_loss_function=sampled_loss)

        self.cost  = tf.reduce_sum(loss)/self.batch_size
        self.optim = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        tf.summary.scalar("cost", self.cost)

        #self.lr   = tf.Variable(0.0, trainable=False)
        #tvars     = tf.trainable_variables()
        #grads, _  = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def _build_zero_state(self, h):
        c = tf.zeros_like(h)
        return tf.concat(1, [c, h])

    def get_model_dir(self):
        model_name = type(self).__name__ or "Reader"
        return "{}_{}".format(model_name, self.batch_size), model_name


    def save(self, sess):
        self.saver = tf.train.Saver()
        print("Saving checkpoints...")
        model_dir, model_name = self.get_model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name))

    def load(self, sess, checkpoint_dir):
        model_dir, model_name = self.get_model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        self.saver = tf.train.Saver()

        print("Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        return False

    def sample(self, sess, inputs, rev_vocab):
        x_hats = []
        for x in inputs:
            logits    = sess.run(self.logits, {self.input_data: [x]})
            softmaxes = sess.run([tf.nn.softmax(logit) for logit in logits])
            idxs      = [np.argmax(softmax) for softmax in softmaxes]
            chars     = [rev_vocab[i] for i in idxs]
            x_hats.append(chars)
        return x_hats

    def train(self, sess, data_path, iterator, iterations=100, save_iters=10):
        model_dir, model_name = self.get_model_dir()
        merged_sum = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/{}".format(model_dir),
                                        sess.graph)

        sess.run(tf.initialize_all_variables())
        i = 0
        for x,y in iterator(data_path, self.vocab, self.sent_steps,
                            self.doc_steps, batch_size=self.batch_size):
            outputs = sess.run(self.logits + [self.optim, merged_sum],
                               {self.input_data: x, self.output_data: y})
            logits      = outputs[:-2]
            summary_str = outputs[-1]

            if i % 2 == 0:
                writer.add_summary(summary_str, i)

            if i % save_iters == 0 and i != 0:
                self.save(sess)

            if i == iterations: return

            if i % 10 == 0:
                print("Iteration: {}".format(i))

            i += 1
