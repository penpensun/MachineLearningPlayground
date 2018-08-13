import numpy as np;
import tensorflow as tf;

rand_matrix = np.random.randint(low = 0, high = 10, size = (3,4,5),dtype = np.int8);
rand_matrix_tolist = rand_matrix.tolist();
print("typeof rand_matrix: ", type(rand_matrix));
rand_matrix_var = tf.Variable(initial_value = rand_matrix);
print("rand_matrix_var:  ", rand_matrix_var);
concat_matrix = tf.concat(values = rand_matrix_tolist, axis = 1);
print("concat_matrix:  ",concat_matrix);

concat_matrix_2 = tf.concat(values = rand_matrix_tolist, axis = 0);
print("concat_matrix_2: ", concat_matrix_2);
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    concat_matrix_run = sess.run(concat_matrix);
    print(concat_matrix_run);
    print(concat_matrix.shape);
    print("concat_matrix_2");
    concat_matrix_run_2 = sess.run(concat_matrix_2);
    print(concat_matrix_run_2);


t1 = [[1,2,3],[4,5,6]];
t2 = [[7,8,9],[10,11,12]];
print(type(t1));
print(tf.concat([t1,t2],0));

print(tf.concat([t1,t2],1));

np_concat = np.concatenate(rand_matrix, axis = 0);
print(np_concat);



indices = [0,2,-1,1];
depth = 3;
one_hot_matrix = tf.one_hot(indices,depth,on_value = 5.0, off_value = 0.0, axis =-1);

print(one_hot_matrix);
with tf.Session() as sess:
    one_hot_matrix_run = sess.run(one_hot_matrix);
    print(one_hot_matrix_run);