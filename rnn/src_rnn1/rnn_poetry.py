import collections;
import numpy as np;
import tensorflow as tf;

#data_folder = "";
data_folder = "./data/";

def read_poetrys():
    global peotry_file;
    global poetrys;
    with open(poetry_file, "r", encoding = 'utf-8', ) as f:
        for line in f:
            try:
                title,content = line.strip().split(":");
                content = content.replace(' ',"");
                if '_' in content or '（' in content or '(' in content \
                        or "《" in content or '[' in content:
                    continue;
                if len(content) < 5 or len(content) >79:
                    continue;
                content = '['+content+']';
                poetrys.append(content);
            except Exception as e:
                pass


def preprocessing():
    #sort the poetries with number of characters
    global poetrys;
    poetrys = sorted(poetrys, key = lambda line: len(line));
    print ("Total number of poetries: %d"%len(poetrys));

    #compute the occurrence of each character
    all_words = [];
    for poetry in poetrys:
        all_words += [word for word in poetry];
    counter = collections.Counter(all_words);
    count_pairs = sorted(counter.items(), key = lambda x: -x[1]);
    words, _ = zip (*count_pairs);
    # use the k most frequent words
    words = words[:len(words)]+(' ',);

    # map each word to an id
    word_num_map = dict(zip(words, range(len(words))));
    # Convert the word into one-hot vector
    to_num = lambda word: word_num_map.get(word, len(words));
    poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys];

    return poetrys_vector, word_num_map, words;

'''
poetrys[-20]: [灵矶盘礴兮溜奔错漱，冷风兮镇冥壑。研苔滋兮泉珠洁，一饮一憩兮气想灭。磷涟清淬兮涤烦矶，灵仙境兮仁智归。中有琴兮徽以玉，峨峨汤汤兮弹此曲，寄声知音兮同所欲。]
peotrys_vector[-20]: [2, 369, 1843, 643, 5124, 697, 1879, 1614, 1954, 1815, 0, 438, 7, 697, 1495, 750, 1105, 1, 2901, 385, 1595, 697, 217, 434, 1282, 0, 10, 563, 10, 2782, 697, 192, 545, 934, 1, 2399, 2511, 46, 3551, 697, 2427, 1224, 1843, 0, 369, 159, 863, 697, 1336, 2131, 24, 1, 21, 19, 383, 697, 2010, 539, 82, 0, 1343, 1343, 1715, 1715, 697, 1024, 39, 269, 0, 290, 59, 27, 483, 697, 112, 361, 99, 1, 3]

'''

def gen_batch():
    global batch_size;
    global poetrys_vector;
    global word_num_map;


    x_batches =[];
    y_batches =[];

    for i in range (n_chunk):
        start_index = i*batch_size;
        end_index = start_index+batch_size;

        batches = poetrys_vector[start_index:end_index];
        length = max(map(len,batches));
        xdata = np.full((batch_size, length), word_num_map[' '], np.int32);
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row];

        ydata = np.copy(xdata);
        ydata[:, :-1] = xdata[:, 1:];
        """
        xdata             ydata
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(xdata);
        y_batches.append(ydata);

    return x_batches,y_batches;

def neural_network(model = 'lstm', rnn_size = 128, num_layers =2):
    global words;
    global input_data,output_targets;
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell;
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell;
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell;

    cell = cell_fun(rnn_size, state_is_tuple= True);
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple= True);

    initial_state = cell.zero_state(batch_size, tf.float32);

    with tf.variable_scope('rnnlm', reuse = tf.AUTO_REUSE):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1]);
        softmax_b = tf.get_variable("softmax_b", [len(words)+1]);

        with tf.device("/cpu:0"):
            embeddings = tf.get_variable("embedding", [len(words)+1, rnn_size]);
            inputs = tf.nn.embedding_lookup(embeddings, input_data);

    outputs, last_state = tf.nn.dynamic_rnn(cell = cell, inputs = inputs, initial_state = initial_state, scope = 'rnnlm');
    output = tf.reshape(outputs, [-1, rnn_size]);
    logits = tf.matmul(output, softmax_w)+softmax_b;
    probs = tf.nn.softmax(logits);
    return logits, last_state, probs, cell, initial_state;

def train_neural_network():
    '''Train the neural network'''
    global output_targets;
    global words;
    global n_chunk;
    global x_batches, y_batches;
    global n_chunk;
    logits, last_state, _, _, _ = neural_network();
    targets = tf.reshape(output_targets, [-1]);
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],[targets],[tf.ones_like(targets, dtype = tf.float32)], len(words)
    );
    cost = tf.reduce_mean(loss);
    learning_rate = tf.Variable(0.0, trainable = False);
    tvars = tf.trainable_variables();
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5);
    optimizer = tf.train.AdamOptimizer(learning_rate);
    train_op = optimizer.apply_gradients(zip(grads,tvars));

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver(tf.all_variables());
        for epoch in range(50):
            sess.run(tf.assign(learning_rate, 0.002*(0.97 ** epoch)));
            n = 0;
            for batch in range(n_chunk):
                train_loss, _, _ = sess.run([cost, last_state, train_op], feed_dict= {input_data: x_batches[n], output_targets: y_batches[n]} );
                n += 1;
                #print(epoch, batch, train_loss);
                if(batch % 2000 == 0):
                    print("epoch %d , batch %d is finished"%(epoch, batch));
            if epoch % 7 == 0:
                saver.save(sess, data_folder+'poetry.module', global_step=epoch);
            print("epoch %d finished."%epoch);


def gen_poetry():
    global words;
    global word_num_map;
    global n_chunk;
    def to_word(weights):
        t = np.cumsum(weights);
        s = np.sum(weights);
        sample = int(np.searchsorted(t, np.random.rand(1)*s));
        #print("length of words: %d"%len(words));
        #print("value of sample: %d"%sample);
        return words[sample];

    _, last_state, probs, cell, initial_state = neural_network();

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver(tf.all_variables());
        saver.restore(sess, data_folder+'poetry.module-49');

        state_ = sess.run(cell.zero_state(1, tf.float32));
        x = np.array([list(map(word_num_map.get, '['))]);
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data:x, initial_state: state_});
        word = to_word(probs_);

        poem = '';
        while word != ']':
            poem += word;
            x = np.zeros((1,1));
            x[0,0] = word_num_map[word];
            [probs_, state_] = sess.run([probs, last_state], feed_dict= {input_data:x, initial_state: state_});
            word = to_word(probs_);

        return poem;

if __name__ == '__main__':
    #poetry_file = "c:/workspace/data/poetry.txt";
    #poetry_file = "/Users/penpen926/workspace/data/poetry.txt";
    poetry_file = data_folder+'poetry.txt';
    # Poetry collections
    poetrys = [];
    read_poetrys();
    poetrys_vector, word_num_map,words = preprocessing();

    ''' for test'''
    print(poetrys[-20]);
    print(poetrys_vector[-20]);
    ''' end test code'''
    #Define batch_size,
    batch_size = 1;
    n_chunk = len(poetrys_vector) // batch_size;
    x_batches, y_batches = gen_batch();
    input_data = tf.placeholder(dtype = tf.int32, shape = [batch_size, None]);
    output_targets = tf.placeholder(dtype = tf.int32, shape = [batch_size, None]);
    #train_neural_network();
    print(gen_poetry());



