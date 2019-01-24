# import tushare as ts;
# ts.set_token('e2ada722f9548429c12be8c91f94702d8422f91f3c30f5aa71d08199')
# pro = ts.pro_api();
# df = pro.query('daily', ts_code='601398.SH', start_date='20160101', end_date='20181231');
#print(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

input_file = "/Users/penpen926/workspace/data/stock/601398.csv"
df = pd.read_csv(input_file)[['date', 'close']];

df = df[::-1]
#print(df);

# fig = plt.figure();
# ax = fig.add_subplot(111);
# ax.set_title("ICBC stock price");
# plt.xlabel('Date');
# plt.ylabel('Price');
# ax.scatter(df['date'], df['close'].values)
#plt.show();

#normalization
normalize_data = (df['close'].values - np.mean(df['close'].values))/ np.std(df['close'].values);
normalize_data = normalize_data[:, np.newaxis];
print(normalize_data);
time_step = 20
rnn_unit = 10
batch_size = 60
input_size = 1
output_size = 1
lr = 0.0006
train_x, train_y = [], []
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i+time_step]
    y = normalize_data[(i + 1): i+time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist());

print('typeof train_x: ', type(train_x))
print('train_x: \n', train_x)
print('typeof train_y: ', type(train_y))
print('train_y: \n', train_y);

X = tf.placeholder(tf.float32, [None, time_step, input_size])
Y = tf.placeholder(tf.float32, [None, time_step, output_size]);

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit,])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size,]))
}

def lstm(batch):
    w_in=weights['in']
    b_in=biases['in']
    input = tf.reshape(X, [-1, input_size]);
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit]);
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32);
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32);
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

def train_lstm():
    global batch_size
    pred,_ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])));
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables());
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size
                # 每10步保存一次参数
                if step % 10 == 0:
                    print(i, step, loss_)
                    print("保存模型：", saver.save(sess, './stock.model'))
                step += 1

if (__name__ =='__main__'):
    train_lstm()