import tensorflow as tf;
import numpy as np;

with tf.name_scope("name_scope_x"):
    var1 = tf.get_variable(name='var1', shape=[1],dtype = tf.float32);
    var3 = tf.Variable(name='var2', initial_value=[2], dtype = tf.float32);
    var4 = tf.Variable(name='var2', initial_value=[2], dtype = tf.float32);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print(var1.name,",", sess.run(var1));
    print(var3.name,",", sess.run(var3));
    print(var4.name,",", sess.run(var4));

'''
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype =tf.float32);
    var2 = tf.get_variable(name='var2', shape=[1], dtype= tf.float32);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print(var1.name, ",", sess.run(var1));
    print(var2.name, ",", sess.run(var2));
'''
z = np.random.randint(0, 10, size = [3,2]);
print(z);
y = tf.one_hot(z, 10, on_value = 1, off_value = None, axis=0);
with tf.Session() as sess:
    print(np.shape(sess.run(y)));


y = tf.one_hot(z, 10, on_value = 1, off_value = None);
with tf.Session() as sess:
    print(np.shape(sess.run(y)));
    print(sess.run(y));


embeddings = np.random.rand(3,4);
print("embedding matrix:" );
print(embeddings);
z = np.random.randint(0, 3, size = [5, 2]);
y = tf.nn.embedding_lookup(params = embeddings, ids = z);
print("The input:");
print(z);
print(y.shape);
with tf.Session() as sess:
    print(sess.run(y));
