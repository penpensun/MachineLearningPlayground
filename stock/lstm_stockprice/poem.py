import tensorflow as tf;
import numpy as np;


#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 128);
#inputs = tf.placeholder(dtype=np.float32, shape = [32, 100])
#h0 = lstm_cell.zero_state(batch_size = 32, dtype=np.float32);
#output, h1 = lstm_cell.call(inputs, h0)

print("init rnn cell");
rnnCell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128);
inputs = tf.placeholder(dtype = tf.float32, shape = [32, 100])
h0 = rnnCell.zero_state(batch_size = 32, dtype = tf.float32);
print("compute the hidden state h1");
output, h1 = rnnCell.__call__(inputs, h0);
print(h1);
output, 