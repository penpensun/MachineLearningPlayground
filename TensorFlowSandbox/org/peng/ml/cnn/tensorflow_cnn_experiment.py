# This file is written to carry out some experiment to understand in details
# how tf.nn.conv2d works. What its input looks like and what its output looks like.

import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

def learnConv2d():
    inputTensor = tf.Variable(tf.truncated_normal(shape = [3,10,10,1],mean =0.0, stddev = 0.5));
    inputTensorShape = tf.shape(inputTensor);



    #Construct a conv2d
    weights = tf.Variable(tf.truncated_normal(shape = [2,2,1,5], mean = 0.0, stddev = 0.5));

    conv1 = tf.nn.conv2d(input = inputTensor, filter = weights, strides=[1,1,1,1], padding = "SAME");
    init = tf.global_variables_initializer();
    # Output the conv1
    with tf.Session() as sess:
        sess.run(init);
        print(sess.run(inputTensor));
        print(sess.run(inputTensorShape));
        sess.run(conv1);
        print(conv1.get_shape());

# This method experiments the situation:
# conv_res + bias;
# The dimension of conv_res is [batch, filtered_height, filtered_weight, channel]
# However, the bias only have one dimension: [channel]
# What then is the result of conv_res + bias;
def bias_plus_conv_res():
    conv_res_flattened = tf.Variable([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]);
    conv_res = tf.reshape(conv_res_flattened, shape =[3,3,1,2]);
    bias = tf.constant([1,2], shape = [2]);
    layer_res = conv_res+bias;

    init = tf.global_variables_initializer();
    with tf.Session() as sess:
        sess.run(init);
        print("Bias");
        print(sess.run(bias));
        print("conv_res:");
        print(sess.run(conv_res));
        print("Layer res.");
        print(sess.run(layer_res));

def know_mnist_data():
    #mnist_data = input_data.read_data_sets("/Users/penpen926/workspace/data/data_MNIST", one_hot=True);
    mnist_data = input_data.read_data_sets("C:/workspace/data/data_mnist",one_hot=True);
    print("The type of mnist_data.");
    print(type(mnist_data));
    next_batch = mnist_data.train.next_batch(10);
    print("the type of next batch.");
    print(type(next_batch));
    print(len(next_batch));

    train = mnist_data.train;
    print("The type of training data.")
    print(type(train));

    train_images = mnist_data.train.images;
    print('The type of train images.');
    print(type(train_images));
    print("The shape of train images.");
    print(train_images.shape);

    train_label = mnist_data.train.labels;
    print('The type of training labels.');
    print(type(train_label));
    print("The shape of training labels.");
    print(train_label.shape);

    test = mnist_data.test;
    print('The type of mnist test data.');
    print(type(test));


    print("The first item of the tuple.");
    print(next_batch[0]);
    print("The type of the first item of the tuple.");
    print(type(next_batch[0]));
    print("The shape of the first item of the tuple.");
    print(next_batch[0].shape);


    print("The second item of the tuple.");
    print(next_batch[1]);




#bias_plus_conv_res();

#learnConv2d();
know_mnist_data();