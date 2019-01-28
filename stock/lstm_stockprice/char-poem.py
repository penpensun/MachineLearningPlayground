from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n = 5):
    p = np.squeeze(preds);
    p[np.argsort(p)[:-top_n]] = 0;
    p = p / np.sum(p);
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c;
class CharRnn:
    def __init__(self, num_classes, num_seqs  = 64, num_steps = 50,
                 lstm_size = 128, num_layers=2, learning_rate = 0.001,
                 grad_clip = 5, sampling=False, train_keep_prob=0.5, use_embedding=False,
                 embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                ))