import tensorflow as tf;
import numpy as np;

rnn_cell  = tf.nn.rnn_cell.BasicRNNCell(num_units = 128);
print(rnn_cell.state_size);

inputs = tf.placeholder(np.float32, shape =(32,100));
h0 = rnn_cell.zero_state(32, np.float32);
output, h1 = rnn_cell(inputs, h0);
print(h1.shape);

lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(num_units=128);
inputs = tf.placeholder(np.float32, shape = (32,100));
h0 = lstm_cell.zero_state(32, np.float32);
output, h1 = lstm_cell(inputs, h0);
print(h1.h);
print(h1.c);

def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128);

cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]);

print(cell.state_size); #get (128,128,128)
inputs = tf.placeholder(np.float32, shape=(32,100));
h0 = cell.zero_state(32,np.float32);
output,h1 = cell(inputs,h0);
print(h1);
