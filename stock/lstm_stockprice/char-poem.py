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
                self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets');
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        if self.use_embedding is False:
            self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
        else:
            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size]);
                self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob);
            return drop;
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs = self.lstm_inputs, initial_state = self.initial_state,
                                                                    dtype = tf.float32)



if __name__ == '__main__':
    tensor = tf.constant(value = [
        [[1,2,3,4,5],
         [1,2,3,4,5]],
        [[1,2,3,4,5],
         [1,2,3,4,5]]
    ], dtype = tf.int8);
    tensor_1 = tf.constant(value = [
        [1,2,3,4,5],
        [1,2,3,4,5]
    ])
    tensor_2 = tf.constant(value = [
        [1,2,3,4,5],
        [1,2,3,4,5]
    ])
    tensor_3 = [tensor_1, tensor_2];
    #tensor_concat_1 = tf.concat(tensor, axis = 1)
    tensor_concat_1 = tf.reshape(tensor, shape = [-1, 5])
    tensor_concat_2 = tf.concat(tensor, axis = 0)
    tensor_concat_3 = tf.concat(tensor_3, axis = 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        print('tensor_concat_1:\n', sess.run(tensor_concat_1));
        print('tensor_concat_2:\n', sess.run(tensor_concat_2));
        print('tensor_concat_3:\n', sess.run(tensor_concat_3));
