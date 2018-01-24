'''
Created on 22.12.2017

@author: GGTTF
'''
from __future__ import print_function;
import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;
import requests;
import os;
import pandas as pd;
import matplotlib.pyplot as plt;

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
    
    
def tf_tutorial_1():
    node1 = tf.constant(3.0, dtype= tf.float32);
    node2 = tf.constant(4.0)
    print(node1, node2);


def tf_tutorial_2():
    node1 = tf.constant(3.0, dtype=tf.float32);
    node2 = tf.constant(4.0);
    node3 = tf.add(node1,node2);
    sess = tf.Session();
    print("session run, node1 and node2.");
    print(sess.run([node1,node2]));
    print("node3: ",node3);
    print("sess.run(node3):  ",sess.run(node3));
    
    
def tf_tutorial_3():
    a = tf.placeholder(tf.float32);
    b = tf.placeholder(tf.float32);
    adder_node = a+b;
    sess = tf.Session();
    print("sess.run()1 ",sess.run(adder_node,{a:3, b:4.5}));
    print("sess.run()2 ",sess.run(adder_node,{a:[1,3], b:[2,3]}));
    add_and_triple = adder_node*3;
    print(sess.run(add_and_triple,{a:3,b:4.5}));

def tf_tutorial_4():
    W = tf.Variable([.3],dtype=tf.float32);
    b = tf.Variable([-.3],dtype = tf.float32);
    x = tf.placeholder(tf.float32);
    linear_model = W*x+b;
    sess = tf.Session();
    #init the values
    init = tf.global_variables_initializer();
    sess.run(init);
    print("linear model: ",sess.run(linear_model,{x:[1,2,3,4]}));
    
    y = tf.placeholder(tf.float32);
    squared_error = tf.square(linear_model - y);
    loss = tf.reduce_sum(squared_error);
    print("squared loss: ", sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}));
    
    #Train the model
    optimizer = tf.train.GradientDescentOptimizer(0.01);
    train = optimizer.minimize(loss);
    
    #set the values to init state
    sess.run(init);
    #Train the model
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]});
    
    # Output the train result
    print("The train result: ");
    curr_W, curr_b, curr_loss = sess.run([W,b, loss], {x:[1,2,3,4],y:[0,-1,-2,-3]});
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss));
    
    
def tf_linear_model1():
    os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/");
    filedir = "/Users/penpen926/workspace/MachineLearningPlayground/data/";
    filename = '/ex1data1.txt';
    data = pd.read_csv(filedir+filename,header = None, names = ['population','profit']);
    
    #Peform normalization 
    data = (data-data.mean())/data.std();
    
    #Use cnn to train the linear model
    preds = tf.placeholder(tf.float32);
    obs = tf.placeholder(tf.float32);
    w = tf.Variable(0.0,dtype = tf.float32);
    b = tf.Variable(0.0,dtype = tf.float32);
    model = w*preds+b;
    sess= tf.Session();
    squared_error = tf.square(model-obs);
    loss = tf.reduce_sum(squared_error);
    
    #set init node
    init = tf.global_variables_initializer();
    
    #set the optimizer
    opt = tf.train.GradientDescentOptimizer(0.01);
    #set train node
    train = opt.minimize(loss);
    
    #print type of dataframe indexing
    preds_list = data['population'].tolist();
    obs_list = data['profit'].tolist();
    
    #start running
    sess.run(init);
    #train the model
    for i in range(1000):
        sess.run(train, {preds: preds_list, obs:obs_list });
    
    trained_w,trained_b,loss = sess.run([w,b,loss],{preds: preds_list,obs:obs_list});
    print('After training.');
    print("w: %s, b: %s, loss: %s"%(trained_w, trained_b,loss));
    
    pred_price = trained_w*preds;
    pred_price_list  = sess.run(pred_price, {preds: preds_list});
    print(pred_price_list);
    
    fig,ax = plt.subplots(figsize=(12,8));
    ax.plot(preds_list, pred_price_list,'r',label="Predictions");
    
    ax.scatter(data.population, data.profit, label="Training data");
    ax.legend(loc=2);
    ax.set_xlabel('Population');
    ax.set_ylabel('Profit');
    ax.set_title('Predicted Profit vs. Population Size');
    plt.savefig("tensorflow_plot.png");
    plt.show();
        

def tf_multiple_linear_regression1():
    os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/");
    filedir = "/Users/penpen926/workspace/MachineLearningPlayground/data/";
    filename = '/ex1data2.txt';
    data = pd.read_csv(filedir+filename,names = ['size','bedrooms','price'],header = None);
    #Normalizatin of the data
    data = (data - data.mean())/data.std();
    
    #set the preds and obs
    pred_size_list = data['size'].tolist();
    pred_bedrooms_list = data['bedrooms'].tolist();
    obs = data['price'].tolist();
    
    #set up the tensors
    x_size = tf.placeholder(dtype = tf.float32);
    x_bedroom = tf.placeholder(dtype=tf.float32);
    y_obs = tf.placeholder(dtype =tf.float32);
    w_size = tf.Variable(0.0, dtype=tf.float32);
    w_bedroom = tf.Variable(0.0, dtype =tf.float32);
    b = tf.Variable(0.0);
    init = tf.global_variables_initializer()
    
    #set up the model
    model = w_size*x_size + w_bedroom*x_bedroom+b;
    #set up the loss function
    square_error = tf.square(model-y_obs);
    loss = tf.reduce_sum(square_error);
    #set up the training model
    train_model = tf.train.GradientDescentOptimizer(0.01);
    train = train_model.minimize(loss);
    
    sess = tf.Session();
    #Init
    sess.run(init);
    
    for i in range(1000):
        sess.run(train, {x_size:pred_size_list, x_bedroom: pred_bedrooms_list, y_obs: obs});
    
    trained_w_size, trained_w_bedroom, trained_b, loss = sess.run([w_size, w_bedroom, b,loss], {x_size:pred_size_list,x_bedroom: pred_bedrooms_list, y_obs:obs});
    print("parameter w_size: %s, parameter w_bedroom: %s, b: %s, loss: %s"\
          %(trained_w_size,trained_w_bedroom, trained_b, loss));
    


def tf_estimator_learn1():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])];
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns);
    x_train = np.array([1.,2.,3.,4.]);
    y_train = np.array([0.,-1.,-2.,-3.]);
    
    x_eval = np.array([2.,5.,8.,1.]);
    y_eval = np.array([-1.01,-4.1,-7,0.]);
    
    input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True);
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True);
    eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True);
    estimator.train(input_fn=input_fn,steps=1000);
    train_metrics = estimator.evaluate(input_fn=train_input_fn);
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn);
    print("train metrics: %r"%train_metrics);
    print("eval mertrics: %r"%eval_metrics);
    
#tf_feed();
#tf_mnist_softmax();
#test_http_request();
    
#tf_tutorial_1();
#tf_tutorial_2();
#tf_tutorial_3();
#tf_tutorial_4()    


#tf_linear_model1();
#tf_multiple_linear_regression1();     
tf_estimator_learn1();