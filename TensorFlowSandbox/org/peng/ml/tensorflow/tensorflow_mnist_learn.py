
#This file contains the self-written code to learn conv nn on mnist

import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

mnist = input_data.read_data_sets("/Users/penpen926/workspace/data/MNIST_data", one_hot= True);


def variable_weights(shape):
    return tf.Variable(tf.truncated_normal(shape = shape, mean = 0.0, dtype = tf.float32, stddev = 0.1));

def variable_bias(shape):
    return tf.Variable(tf.constant(0.1,shape = shape, dtype = tf.float32));

def conv2d(input,weights ):
    return tf.nn.conv2d(input = input, filter = weights,
                        strids = [1,1,1,1], padding = "SAME");
def max_2x2_pooling(input):
    return tf.nn.max_pool(value = input,ksize = [1,2,2,1], strides = [1,1,1,1],padding ="SAME");

def trainData():
    x = tf.placeholder(tf.float32,shape=[None, 784]);
    y = tf.placeholder(tf.float32,shape = [None, 10]);

    x_image = tf.reshape(tensor = x, shape = [-1,28,28,1]);

    # Construct the first conv layer
    shape_conv1_variable = tf.constant([5,5,1,32]);
    conv1_weights = variable_weights(shape_conv1_variable);
    #conv1_bias = variable_bias([28*28, 32]);
    conv1_bias = variable_bias([32]);
    conv1_res = conv2d(x_image, conv1_weights)+conv1_bias;

    conv1_res_relu = tf.nn.relu(conv1_res);

    # First pooling layer
    pool1_res = max_2x2_pooling(conv1_res_relu);

    # Second conv layer
    shape_conv2_variable = tf.constant([5,5,32,64]);
    conv2_weights = variable_weights(shape_conv2_variable);
    #conv2_bias = variable_bias([14*14,64]);
    conv2_bias = variable_bias([64]);
    conv2_res = conv2d(pool1_res, conv2_weights)+conv2_bias;
    conv2_res_relu = tf.nn.relu(conv2_res);

    # Second pooling layer
    pool2_res = max_2x2_pooling(conv2_res_relu);

    # First full conn layer
    fc1_weights = variable_weights([7*7*64,1024]);
    fc1_bias = variable_bias([1024]);
    pool2_res_flattened = tf.reshape(pool2_res,shape=[-1,7*7*64]);

    fc1_res = tf.matmul(pool2_res, fc1_weights)+fc1_bias;

    # dropout layer
    dropout1 = tf.nn.dropout(fc1_res, keep_prob = 0.25);

    # Second full conn layer
    fc2_weights = variable_weights([1024, 10]);
    fc2_bias = variable_bias([10]);

    readout = tf.matmul(dropout1, fc2_weights)+fc2_bias;

    # Get the readout number
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = readout));
    trainer = tf.train.AdamOptimizer(1e-4).minimize(loss);

    # Get the accuracy after training.
    correct_predictions = tf.equal(tf.argmax(readout, axis = 1), tf.argmax(y, axis =1));
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32));

    with tf.Sesssion() as sess:
        sess.run(tf.global_variables_initializer());
        for i in range (1000):
            batch = mnist.train.next_batch(50);
            

