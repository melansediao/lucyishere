from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
from collections import Counter


_PAD = b"_PAD"
_EOS = b"_EOS"
_EOD = b"_EOD"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _EOS, _EOD, _UNK]

PAD_ID = 0
EOS_ID = 1
EOD_ID = 2
UNK_ID = 3


def pre_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[(max_len - len(lst)):] = lst
    return nlst


def post_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[:len(lst)] = lst
    return nlst


def load_dictionary(dictionary_path):
    with gfile.GFile(dictionary_path, mode="rb") as f:
        rev_vocab = [line.strip() for line in f]

    vocab = {k: i for i,k in enumerate(rev_vocab)}
    return vocab, rev_vocab


#def initialize_vocabulary(data_path, max_vocab):
#    with gfile.GFile(data_path, mode="rb") as f:
#        lines  = [line.split() for line in f]
#        tokens = [item for sublist in lines for item in sublist]
#        counts = Counter(tokens)
#
#    counts = sorted(counts.iteritems(), key=lambda x: x[1],
#                    reverse=True)[:max_vocab-len(_START_VOCAB)]
#    top_vocab = set(map(lambda x: x[0], counts))
#    vocab = {k:i for i,k in enumerate(top_vocab)}
#    i = len(vocab)
#    for k in _START_VOCAB:
#        vocab[k] = i
#        i += 1
#
#    return vocab


def initialize_vocabulary(vocab_path):
    if gfile.Exists(vocab_path):
        rev_vocab = []
        with gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file {} not found.".format(vocab_path))


def create_vocabulary(vocab_path, data_path, max_vocab):
    max_sent_len = 0
    max_doc_len = 0

    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
        doc_len = 0
        for line in f:
            if line.strip() == "":
                if doc_len > max_doc_len:
                    max_doc_len = doc_len
                doc_len = 0
            else:
                doc_len += 1

                sent = line.split()
                sent_len = len(sent)
                if sent_len > max_sent_len:
                    max_sent_len = sent_len

                for w in sent:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
        if doc_len > max_doc_len:
            max_doc_len = doc_len

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocab:
            vocab_list = vocab_list[:max_vocab]
        with gfile.GFile(vocab_path, mode="wb") as f:
            for w in vocab_list:
                f.write(w + b"\n")

    return max_sent_len + 1, max_doc_len + 1


#def get_max_sent_doc(data_path):
#    max_sent_len = 0
#    max_doc_len = 0
#    with gfile.GFile(data_path, mode="rb") as f:
#        doc_len = 0
#        for line in f:
#            doc_len += 1
#            if line.strip() == "":
#                if doc_len > max_doc_len:
#                    max_doc_len = doc_len
#
#                doc_len = 0
#
#            sent_len = len(line.split())
#            if sent_len > max_sent_len:
#                max_sent_len = sent_len
#
#    # add one for EOS, EOD tokens
#    return max_sent_len + 1, max_doc_len + 1


def data_iterator(data_path, vocab, max_sent_len, max_doc_len, batch_size=1):
    """
    Takes in target path instead of source and then reverses tokens itself.
    """

    with gfile.GFile(data_path, mode="rb") as f:
        doc_n   = 0
        doc     = []
        inputs  = []
        outputs = []
        for line in f:
            if line.strip() == "":
                #doc = [pre_pad(sent + [vocab["_EOS"]], vocab["_PAD"], max_sent_len)
                #       for sent in doc]
                #doc.append([vocab["_EOD"]]*max_sent_len)
                #doc = pre_pad(doc, [vocab["_PAD"]]*max_sent_len, max_doc_len)
                #yield [[sent[::-1] for sent in doc[::-1]]], [doc]
                #doc = []

                doc.append([vocab["_EOD"]]*max_sent_len)
                doc = post_pad(doc, [vocab["_PAD"]]*max_sent_len, max_doc_len)
                input = [post_pad(sent[::-1], vocab["_PAD"], max_sent_len)
                         for sent in doc[::-1]]
                output = [post_pad(sent, vocab["_PAD"], max_sent_len)
                          for sent in doc]

                doc_n += 1
                inputs.append(input)
                outputs.append(output)
                if doc_n == batch_size:
                    yield inputs, outputs
                    inputs  = []
                    outputs = []
                    doc_n   = 0

                doc = []
            else:
                doc.append([vocab[i] if i in vocab else vocab["_UNK"]
                            for i in line.split()] + [vocab["_EOS"]])
