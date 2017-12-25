'''
Created on 22.12.2017

@author: GGTTF
'''
import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

import requests;

def tf1():
    xData = np.float32(np.random.rand(2,100));
    yData = np.float32(np.random.rand(2,100));
    print(xData);
    
    
    b  = tf.Variable(tf.zeros([1]));
    W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0));
    y = tf.matmul(W,xData)+b
    
    loss = tf.reduce_mean(tf.square(y-yData));
    optimizer = tf.train.GradientDescentOptimizer(0.5);
    train = optimizer.minimize(loss);
    
    init = tf.initialize_all_variables();
    
    sess = tf.Session();
    sess.run(init);
    
    for step in range(0,201):
        sess.run(train);
        if step %20 == 0:
            print (step,sess.run(W), sess.run(b));
        


def tf2():
     matrix1 = tf.constant([[3.,3.]]);
     matrix2 = tf.constant([[2.],[2.]]);
     
     product = tf.matmul(matrix1,matrix2);
     sess =tf.Session();
     result = sess.run(product);
     
     print(result);
     
     
     with tf.Session() as sess:
         result = sess.run([product]);
         print(result);
         
         

def tf3():
    sess = tf.InteractiveSession();
    x = tf.Variable([1.0, 2.0]);
    a = tf.constant([3.0, 3.0]);
    
    x.initializer.run();
    
    sub = tf.subtract(x,a);
    
    print(sub.eval());
    


def tf4():
    state = tf.Variable(0, name=  "counter");
    one = tf.constant(1);
    new_value = tf.add(state, one);
    update = tf.assign(state,new_value);
    
    init_op = tf.initialize_all_variables();
    with tf.Session() as sess:
        sess.run(init_op);
        print(sess.run(state));
        for i in range(3):
            sess.run(update);
            print(sess.run(state));
    
    

def tf5():
    input1 = tf.constant(3.0);
    input2 = tf.constant(2.0);
    input3 = tf.constant(5.0);
    intermed = tf.add(input2, input3);
    mul = tf.multiply(input1, intermed);
    
    with tf.Session() as sess:
        result = sess.run([mul, intermed]);
        print(result);
    
def tf_feed():
    input1 = tf.placeholder(tf.float32);
    input2 = tf.placeholder(tf.float32);
    output = tf.multiply(input1,input2);
    
    with tf.Session() as sess:
        print("Result:");
        print(sess.run([output],feed_dict={input1:[7.0], input2:[2.0]} ) );
    



def tf_mnist_softmax():
    mnist = input_data.read_data_sets("c:/workspace/data/",one_hot=True);
    print(mnist);
    
    
def test_http_request():
    r = requests.get("https://github.com/timeline.json");
    print(r);

#tf_feed();
#tf_mnist_softmax();
test_http_request();
    


     