from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os



# BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
BasicLSTMCell = tf.nn.rnn_cell.LSTMCell 

class HierarchicalAutoencoder(object):
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
        print ("build_model ...")
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
        print ("embedding...")
        with tf.device("/gpu:0"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.size])
            self.inputs = []
            for s in range(self.doc_steps):
                print (str(s)+"/"+str(self.doc_steps))
                self.inputs.append(tf.nn.embedding_lookup(self.embedding,
                                                          self.input_data[:,s,:]))

        print ("encode_word_cell ...")
        # ENCODE
        # TODO: make this multi-layer
        self.encode_word_cell = BasicLSTMCell(self.size)
        self.encode_sent_cell = BasicLSTMCell(self.size)

        word_state   = self.encode_word_cell.zero_state(self.batch_size, tf.float32)
        sent_encodes = []
        sent_state   = self.encode_sent_cell.zero_state(self.batch_size, tf.float32)

        print ("encode_RNN_word ...")
        with tf.variable_scope("encode_RNN_word"):
            for s in range(self.doc_steps):
                for t in range(self.sent_steps):
                    print ("encode_RNN_word ...",str(s)+"/"+str(self.doc_steps)+" "+str(t)+"/"+str(self.sent_steps))
                    if t > 0 or s > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, word_state) = self.encode_word_cell(self.inputs[s][:,t,:], word_state)
                sent_encodes.append(cell_output)

        print ("encode_RNN_sentence ...")
        with tf.variable_scope("encode_RNN_sentence"):
            for s in range(self.doc_steps):
                print ("encode_RNN_sentence ...",str(s)+"/"+str(self.doc_steps))
                
                if s > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, sent_state) = self.encode_sent_cell(sent_encodes[s],
                                                                  sent_state)
            doc_encode = cell_output

        self.doc_encode = doc_encode

        # DECODE
        self.decode_sent_cell = BasicLSTMCell(self.size)
        self.decode_word_cell = BasicLSTMCell(self.size)

        sent_decode  = tf.zeros((self.batch_size, self.size))  # 2,16

        print ("doc_encode",doc_encode)
        sent_state   = self._build_zero_state(doc_encode)
        print ("sent_state",sent_state)


        self.logits  = []
        word_decodes = []

        print ("decode_RNN_sentence ...")
        with tf.variable_scope("decode_RNN_sentence"):
            for s in range(self.doc_steps):
                print (str(s)+"/"+str(self.doc_steps))
                if s > 0: tf.get_variable_scope().reuse_variables()
                
                print ("sent_decode",sent_decode)
                print ("sent_state",sent_state)
                (cell_output, sent_state) = self.decode_sent_cell(sent_decode,sent_state)

                sent_decode = cell_output

                word_decode = tf.zeros((self.batch_size, self.size)) # 2,16
                word_state  = self._build_zero_state(sent_decode)

                with tf.variable_scope("decode_RNN_word"):
                    for t in range(self.sent_steps):
                        print ("decode_RNN_sentence ...",str(s)+"/"+str(self.doc_steps)+" "+str(t)+"/"+str(self.sent_steps))

                        # TODO: issue with get_variable_scope for sentence params?
                        # scope here is decode_RNN_sentence/decode_RNN_word/
                        if t > 0 or s > 0: tf.get_variable_scope().reuse_variables()
                        (cell_output, word_state) = self.decode_word_cell(word_decode,
                                                                          word_state)
                        word_decode = cell_output

                        word_decodes.append(word_decode)
                        #w = tf.get_variable("w", [self.size, self.vocab_size])
                        #b = tf.get_variable("b", [self.vocab_size])
                        #logit = tf.matmul(cell_output, w) + b
                        #self.logits.append(logit)

                    sent_state = word_state

        w = tf.get_variable("w", [self.size, self.vocab_size])
        b = tf.get_variable("b", [self.vocab_size])

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b,
                                              labels, logits, self.num_samples,
                                              self.vocab_size)

        targets = tf.reshape(self.output_data, [self.batch_size,
                                                self.doc_steps*self.sent_steps])
        targets = [targets[:,i] for i in range(self.doc_steps*self.sent_steps)]
        weights = [tf.ones([self.batch_size]) for _ in range(self.doc_steps*self.sent_steps)]
        # loss    = tf.nn.seq2seq.sequence_loss_by_example(word_decodes, targets, weights,
        #                                                  softmax_loss_function=sampled_loss)

        loss    = tf.contrib.legacy_seq2seq.sequence_loss_by_example(word_decodes, targets, weights,
                                                         softmax_loss_function=sampled_loss)

        self.cost  = tf.reduce_sum(loss)/self.batch_size
        self.optim = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        tf.contrib.deprecated.scalar_summary("cost", self.cost)

        #self.lr   = tf.Variable(0.0, trainable=False)
        #tvars     = tf.trainable_variables()
        #grads, _  = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def _build_zero_state(self, h):
        # TODO: incorporate batch_size
        c = tf.zeros_like(h)
        # return tf.concat([c, h],1 )
        return (c,h)

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
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(model_dir),
                                        sess.graph)

        sess.run(tf.initialize_all_variables())
        i = 0
        for x,y in iterator(data_path, self.vocab, self.sent_steps,
                            self.doc_steps, batch_size=self.batch_size):
            outputs = sess.run(self.logits + [self.optim, merged_sum],
                               {self.input_data: x, self.output_data: y})
            print (i)
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
