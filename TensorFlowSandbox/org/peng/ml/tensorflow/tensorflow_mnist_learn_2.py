import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

mnist = input_data.read_data_sets("c:/workspace/data/data_mnist/", one_hot = True);

def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1));

def bias_variables(shape):
    return tf.Variable(tf.ones(shape));

def conv2d(input, filter):
    return tf.nn.conv2d(input = input, filter = filter, padding = "SAME", strides =[1,1,1,1]);


def max_2x2_pooling(input):
    return tf.nn.max_pool(value = input, ksize = [1,2,2,1], strides =[1,2,2,1], padding = "SAME");


def train_model():
    #Create the model
    x = tf.placeholder(dtype = tf.float32, shape = [None, 784]);
    y = tf.placeholder(dtype = tf.float32, shape = [None, 10]);
    #Reshape the input data
    x_images = tf.reshape(x, shape = [-1, 28, 28, 1]);
    #First convolutional layer
    shape_conv1_filter = [5,5,1,32];
    weights_conv1_filter = weight_variables(shape_conv1_filter);
    bias_conv1 = bias_variables([32]);
    #Get the convoluted output
    conv1 = conv2d(x_images, weights_conv1_filter)+ bias_conv1;
    conv1_relu= tf.nn.relu(features = conv1);

    #First pooling layer
    conv1_pooled = max_2x2_pooling(conv1_relu);

    #Second convolutional layer
    shape_conv2_filter = [5,5,32,64];
    weights_conv2_filter = weight_variables(shape_conv2_filter);
    bias_conv2 = bias_variables([64]);
    #Get the convoluted output
    conv2 = conv2d(conv1_pooled, weights_conv2_filter)+ bias_conv2;
    conv2_relu= tf.nn.relu(features = conv2);

    #Second pooling layer
    conv2_pooled = max_2x2_pooling(input = conv2_relu);

    #First fully connected layer
    shape_fc1 = [7*7*64, 1024];
    weights_fc1 = weight_variables(shape_fc1);
    bias_fc1 = bias_variables([1024]);

    input_fc1 = tf.reshape(conv2_pooled, [-1, 7*7*64]);

    #Get the output from fc1
    fc1 = tf.matmul(input_fc1, weights_fc1)+bias_fc1;
    fc1_relu = tf.nn.relu(fc1);

    #Drop out layer
    keep_prob = tf.placeholder(tf.float32);
    dropout = tf.nn.dropout(x = fc1_relu, keep_prob = keep_prob);
    
    # Second fully connected layer
    shape_fc2 = [1024,10];
    weights_fc2 = weight_variables(shape_fc2);
    bias_fc2 = bias_variables([10]);
    # This is the readout from the cnn
    readout = tf.matmul(dropout, weights_fc2)+bias_fc2;

    # Define the loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = readout));

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss);

    # Get the corrected prediction
    correct_predicitons = tf.equal(tf.argmax(readout, axis = 1), tf.argmax(y, axis=1));
    accuracy = tf.reduce_mean(tf.cast(correct_predicitons,dtype = tf.float32));
    #Train the model in iterative manner
    with tf.Session() as sess:
        #Init all variables
        sess.run(tf.global_variables_initializer());
        for i in range(4000):
            #train the model
            #Get the 50 batch
            next_batch = mnist.train.next_batch(50);
            sess.run(optimizer, feed_dict={x: next_batch[0], y: next_batch[1], keep_prob: 0.75});
            if i%100 == 0:
                #Compute the accuracy:
                print('iteration: ',i,"  accuracy:  ",sess.run(accuracy, \
                                                               feed_dict={x: next_batch[0], y: next_batch[1], keep_prob: 1.0}))
        print('test on test data set.');
        test_batch = mnist.test;
        print(sess.run(accuracy, feed_dict={x: test_batch.images, y: test_batch.labels, keep_prob: 1.0}))

train_model();







