import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

#Read in mnist data
mnist = input_data.read_data_sets("c:/workspace/data/data_mnist/", one_hot = True);

#The function creating a cnn
def conv2d(input, filter):
    return tf.nn.conv2d(input = input, filter = filter, strides = [1,1,1,1], padding = "SAME");

# This function returns randomized weights, serving as the initial set of the weights
def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape = shape, mean = 0.0, stddev = 0.1));

# This function returns the radonmized biases.
def bias_variables(shape):
    return tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = shape));

#This function defines a max 2x2 pooling layer
def max_2x2_pooling(input):
    return tf.nn.max_pool(value = input, ksize = [1,2,2,1], strides=[1,2,2,1], padding = "SAME");

# This function defines the model and trains the model on the mnist training data set.
def train_model():
    # reshape the input
    shape_input = [-1, 28,28,1];
    x = tf.placeholder(dtype = tf.float32, shape = [None, 784]);
    y = tf.placeholder(dtype = tf.float32, shape = [None, 10]);

    x_input = tf.reshape(x, shape_input);

    # first conv layer
    shape_filter_conv1 = [5,5,1,32];
    # Gernate randomized variables
    filter_conv1 = weight_variables(shape_filter_conv1);
    # Gernerate randomized biases
    bias_conv1 = bias_variables([32]);
    conv1 = conv2d(input= x_input, filter = filter_conv1)+bias_conv1;
    # fist relu layter
    conv1_relu = tf.nn.relu(features = conv1);

    #first pooling layer
    pool1 = max_2x2_pooling(conv1_relu);

    #Second conv layer
    shape_filter_conv2 = [5,5,32,64];
    #Generate randomized variables
    filter_conv2 = weight_variables(shape_filter_conv2);
    #Generate randomized biases
    bias_conv2 = bias_variables([64]);
    conv2 = conv2d(input = pool1, filter = filter_conv2)+bias_conv2;
    #second relu
    conv2_relu= tf.nn.relu(features = conv2);
    #second pooling layer
    pool2 = max_2x2_pooling(conv2_relu);

    #First densely connected layer
    shape_fc1 = [7*7*64, 1024];
    # generate randomized variables
    filter_fc1 = weight_variables(shape_fc1);
    # gerenate biass
    biases_fc1 = bias_variables([1024]);

    # The shape for fc1 input
    shape_fc1_input = [-1, 7*7*64];
    # reshape the nn tensor
    fc1_input = tf.reshape(pool2, shape_fc1_input);
    fc1 = tf.matmul(fc1_input,filter_fc1)+biases_fc1;


    #the dropout layer
    keep_prob = tf.placeholder(tf.float32);
    dropout = tf.nn.dropout(x = fc1, keep_prob = keep_prob);

    #The second completely connected layer
    # Generate randomized vaiables
    shape_weights_fc2 = [1024, 10];
    weights_fc2 = weight_variables(shape_weights_fc2);

    #Generate randomized biases
    biases_fc2 = bias_variables([10]);
    fc2 = tf.matmul(dropout, weights_fc2)+biases_fc2;

    # Define loss function and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = fc2));

    # Define the optimizer
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss);
    # Get the correct prediction list
    correct_predictions = tf.equal(tf.argmax(y, axis = 1), tf.argmax(fc2, axis=1));

    # Get the acuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32));

    #Start training the model
    with tf.Session() as sess:
        print("init all variables.");
        sess.run(tf.global_variables_initializer());
        print("Start training the model.");
        for i in range(1000):
            next_batch = mnist.train.next_batch(100);
            sess.run(opt, feed_dict={x: next_batch[0], y: next_batch[1], keep_prob: 1.0});
            if i%100 == 0:
                print("iteration: ", i);
                print('accuracy: ',sess.run(accuracy, feed_dict={x: next_batch[0], y: next_batch[1], keep_prob: 1.0}));


train_model();

