'''
Created on 2017年12月27日

@author: penpen926
'''

# In this module, I play and test the convolution in tensorflow
#http://blog.csdn.net/u012609509/article/details/71215859

from __future__ import division;
import tensorflow as tf;
import numpy as np;
import math;
import pandas as pd;


input_arr = np.zeros((12,15), dtype=np.float32);

number = 0;
for row_idx in range(input_arr.shape[0]):
    for col_idx in range(input_arr.shape[1]):
        input_arr[row_idx][col_idx] = number;
        number+=1;
        
#print("input_arr:")
#print(input_arr);

number = 6;
w_arr = np.zeros((2,3),dtype= np.float32);
for row_idx in range(w_arr.shape[0]):
    for col_idx in range(w_arr.shape[1]):
        w_arr[row_idx][col_idx] = number;
        number-=1;

print("w_arr:")   
print(w_arr);
print(w_arr.shape);

input_arr = np.reshape(input_arr, [1,input_arr.shape[0],input_arr.shape[1],1]);
w_arr = np.reshape(w_arr, [w_arr.shape[0], w_arr.shape[1],1,1]);

print("after reshaping.");
#print("input_arr.");
#print(input_arr);
print("w_arr");
print(w_arr);
print(type(w_arr));
print(w_arr.shape);
print("first element of w_arr.");
print(w_arr[0]);
print("shape of the first element of w_arr.");
print(w_arr[0].shape);

print("second element of w_arr.");
print(w_arr[1]);
print("shape of the second element of w_arr.");
print(w_arr[1].shape);


net_in = tf.constant(value=input_arr, dtype = tf.float32);
w = tf.constant(value=w_arr, dtype=tf.float32);

stride = [1,1,1,1];

result_conv_valid = tf.nn.conv2d(net_in, w,stride,'VALID',True);
result_conv_same = tf.nn.conv2d(net_in, w , stride,'SAME',True);
sess = tf.Session();
sess.run(tf.global_variables_initializer());
valid_conv_res = sess.run(result_conv_valid);
same_conv_res = sess.run(result_conv_same);

sess.close();

#print("valid_conv_res");
#print(valid_conv_res);

#print("same_conv_res");
#print(same_conv_res);


