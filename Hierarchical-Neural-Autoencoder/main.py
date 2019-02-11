from __future__ import division
from __future__ import print_function

from models.hier_autoencoder import HierarchicalAutoencoder as HAE
from paper_data_utils import create_vocabulary
from paper_data_utils import initialize_vocabulary
from paper_data_utils import data_iterator

import tensorflow as tf
import numpy as np


data_path = "paper_data/train_target_permute_segment.txt"
sample_data_path = "paper_data/sample_text.txt"
debug_data_path = "paper_data/debug_text.txt"
size = 16


def main():
    with tf.Session() as sess:
        #max_sent_len, max_doc_len = create_vocabulary("data/vocab",
        #                                              "data/sample.txt", 1000)
        #vocab, rev_vocab = initialize_vocabulary("data/vocab")

        max_sent_len, max_doc_len = create_vocabulary("paper_data/vocab",
                                                      debug_data_path, 50)
        vocab, _ = initialize_vocabulary("paper_data/vocab")

        model = HAE(vocab, max_sent_len, max_doc_len, size=size, batch_size=2)
        sess.run(tf.initialize_all_variables())

        print("max_sent_len=" + str(max_sent_len))
        print("max_doc_len="  + str(max_doc_len))
        #print("len(rev_vocab)=" + str(len(rev_vocab)))
        model.train(sess, debug_data_path, data_iterator, iterations=500, save_iters=25)


def sample():
    with tf.Session() as sess:
        max_sent_len, max_doc_len = 74, 16
        vocab, rev_vocab = initialize_vocabulary("paper_data/vocab")

        model = HAE(vocab, max_sent_len, max_doc_len, size=size)
        model.load(sess, "checkpoint")

        for i,(x,y) in enumerate(data_iterator(debug_data_path, vocab,
                                               max_sent_len, max_doc_len)):
            x_hat = model.sample(sess, x, rev_vocab)
            print(x)
            print(x_hat)
            break

if __name__ == '__main__':
    main()
