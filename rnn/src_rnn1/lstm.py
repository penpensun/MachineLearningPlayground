from __future__ import print_function;
import numpy as np;
import random;
import string;
import tensorflow as tf;
import zipfile;
from six.moves import range;
from src_rnn1.word_embedding import maybe_download;


class BatchGenerator (object):
    _text = None;
    _text_size = None;
    _num_unrollings = None;
    _cursor = None;
    _last_batch = None;
    _vocabulary_size = None;

    def __init__ (self, text, batch_size, num_unrollings, vocabulary_size):
        self._text = text;
        self._text_size = len(text);
        self._batch_size = batch_size;
        self._num_unrollings = num_unrollings;
        self._vocabulary_size = vocabulary_size;
        segment = self._text_size // batch_size;
        self._cursor = [offset * segment for offset in range(batch_size)]

        self._last_batch = self._next_batch();

    def _next_batch(self):
        '''Generate a single batch from the current cursor position in the data'''
        ''' The result will be one-hot vector'''

        batch = np.zeros(shape = (self._batch_size, self._vocabulary_size),dtype = np.float);
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0;
            self._cursor[b] = (self._cursor[b]+1)%self._text_size;
        return batch;

    def next(self):
        '''Generate the next array of batches from the data. the array consists of the last batch
        of the previous array, followed by num_unrollings new ones.'''
        batches = [self._last_batch];
        for step in range(self._num_unrollings):
            batches.append(self._next_batch());
        self._last_batch = batches[-1];
        return batches;

def read_data(filename):
    f = zipfile.ZipFile(filename);
    for name in f.namelist():
        return tf.compat.as_str(f.read(name));
    f.close();

def char2id(char):
    global first_letter;
    if char in string.ascii_lowercase:
        return ord(char) - first_letter+1;
    if char == ' ':
        return 0;
    else:
        print("Unexpected character: %s" %char);
        return 0;

def id2char(dictid):
    global first_letter;
    if dictid >0:
        return chr(dictid + first_letter-1);
    else:
        return ' ';

''' Convert one-hot vector to index, then using id2char to character'''
def characters(probabilities):
    '''Turn a 1-hot encoding or probability distribution over the possible characters back into its (most likely) charactor representation.'''
    return [id2char(c) for c in np.argmax(probabilities,1)];

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string representation"""
    s = ['']*batches[0].shape[0];
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s;

def logprob(predictions, labels):
    '''Log-probability of the true labels in a predicted batch.'''
    predictions[predictions <1e-10] = 1e-10;
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0];

def sample_distribution(distribution):
    '''Sample one element from a distribution assumed to be an array of normalized probabilities'''
    r = random.uniform(0,1);
    s = 0;
    for i in range(len(distribution)):
        s+= distribution[i];
        if s>= r:
            return i;
    return len(distribution) -1;

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape = [1,vocabulary_size], dtype = np.float);
    p[0,sample_distribution(prediction[0])] = 1.0
    return p;

def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size = [1,vocabulary_size]);
    return b/np.sum(b,1)[:,None];



#Definition of the cell computation
# o is h in wikipedia
def lstm_cell(i,o,state):
    global ix,im,ib,fx,fm,fb,cx,cm,cb,ox,om,ob;

    '''Create a LSTm cell. '''
    input_gate = tf.sigmoid(tf.matmul(i,ix)+tf.matmul(o,im)+ib);
    forget_gate = tf.sigmoid(tf.matmul(i,fx)+tf.matmul(o, fm)+fb);
    output_gate = tf.sigmoid(tf.matmul(i,ox)+tf.matmul(o, om)+ob);

    update = tf.matmul(i, cx)+tf.matmul(o,cm)+cb;
    state = forget_gate*state + input_gate* tf.tanh(update);
    output_vector = output_gate*tf.tanh(state)
    return output_vector, state;




if __name__ == '__main__':
    url = 'http://mattmahone4y.net/dc/';
    filename = maybe_download('text8.zip', 31344016);
    text = read_data(filename);
    print("Data zie %d", len(text));
    valid_size = 1000;
    valid_text = text[:valid_size];
    train_text = text[valid_size:];
    train_size = len(train_text);
    print(train_size, train_text[:64]);
    print(valid_size, valid_text[:64]);

    vocabulary_size = len(string.ascii_lowercase) +1; # [a-z] + ' '
    first_letter = ord(string.ascii_lowercase[0]);

    print(char2id('a'), char2id('z'), char2id(' '), char2id('i'));
    print(id2char(1), id2char(26), id2char(0));

    batch_size = 64;
    num_unrollings = 10;

    train_batches = BatchGenerator(text = train_text, batch_size = batch_size,num_unrollings= num_unrollings, vocabulary_size =vocabulary_size);
    valid_batches = BatchGenerator(text = valid_text, batch_size = 1, num_unrollings =1, vocabulary_size = vocabulary_size);

    #Print some batches
    print(batches2string(train_batches.next()));
    print(batches2string(train_batches.next()));
    print(batches2string(valid_batches.next()));
    print(batches2string(valid_batches.next()));

    num_nodes = 64;

    #Define the model
    graph = tf.Graph();
    with graph.as_default():
        # Parameters
        # Input gate: input, previous output and bias
        ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1));
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1));
        ib = tf.Variable(tf.zeros([1, num_nodes]));

        # Forget gate: input, previous output and bias
        fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1));
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1));
        fb = tf.Variable(tf.zeros([1, num_nodes]));

        # Cell state: input, state and bias
        cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1));
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1));
        cb = tf.Variable(tf.zeros([1, num_nodes]));

        # Output gate: input, previous output and bias
        ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1));
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1));
        ob = tf.Variable(tf.zeros([1, num_nodes]));

        # Variables saving state across unrollings
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False);
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False);
        # Classifier weights and biases
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1));
        b = tf.Variable(tf.zeros([vocabulary_size]))

        #Input data
        train_data = list();
        for _ in range(num_unrollings +1):
            train_data.append(
                tf.placeholder(tf.float32, shape = [batch_size, vocabulary_size])
            );
        train_inputs = train_data[:num_unrollings];
        train_labels = train_data[1:]; #labels are inputs shifted by one time step.

        #Unrolled LSTM loop.
        outputs = list();
        output = saved_output;
        state = saved_state;
        for i in train_inputs:
            output, state = lstm_cell(i, output,state);
            outputs.append(output);

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            #Classifier

            logits = tf.nn.xw_plus_b(tf.concat(outputs,0), w,b);
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.concat(train_labels,0))
            );

        #Optimizer
        global_step = tf.Variable(0);
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase= True);
        optimizer = tf.train.GradientDescentOptimizer(learning_rate);
        gradients,v = zip (*optimizer.compute_gradients(loss));
        gradients,_ = tf.clip_by_global_norm(gradients, 1.25);
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step = global_step);

        #Predictions.
        trains_prediciton = tf.nn.softmax(logits);

        #Sampling and validation eval: batch 1, no unrolling
        sample_input = tf.placeholder(tf.float32, shape = [1, vocabulary_size]);
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]));
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]));
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes]))
        );
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state
        );

        with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w,b));

        #Start training
        num_steps = 7001;
        summary_frequency = 100;
        with tf.Session(graph = graph) as sess:
            sess.run(tf.global_variables_initializer());
            print("Initialized.");
            mean_loss = 0;
            for step in range(num_steps):
                batches = train_batches.next();
                feed_dict = dict();
                for i in range(num_unrollings+1):
                    feed_dict[train_data[i]] = batches[i];
                _, l, predictions, lr = sess.run(
                    [optimizer, loss, trains_prediciton, learning_rate], feed_dict = feed_dict);
                mean_loss +=l;
                #print("For test, print the outputs")
               #for i in range(len(outputs)):
                #    print(outputs[i].eval(feed_dict = feed_dict));

                if step % summary_frequency == 0:
                    if step >0:
                        mean_loss = mean_loss/summary_frequency;
                        #The mean loss is an stimate of the loss over the last few batches
                        print(
                            'Average loss at step: %d: %f learning rate: %f' %(step, mean_loss, lr)
                        );
                        mean_loss = 0;
                        labels = np.concatenate(list(batches)[1:]);
                        print('Minibatch perplexity: %0.2f'% float(
                            np.exp(logprob(predictions, labels))
                        ));

                        if step %(summary_frequency*10) == 0:
                            #Generate some samples
                            print('='*80);
                            for _ in range(5):
                                feed = sample(random_distribution());
                                sentence = characters(feed)[0];
                                reset_sample_state.run();
                                for _ in range(79):
                                    prediction = sample_prediction.eval({sample_input: feed});
                                    feed = sample(prediction);
                                    sentence += characters(feed)[0];
                                print(sentence);
                            print('='*80);
                        #Measure validation set perplexity
                        reset_sample_state.run();
                        valid_logprob =0;
                        for _ in range(valid_size):
                            b = valid_batches.next();
                            predictions = sample_prediction.eval({sample_input: b[0]});
                            valid_logprob = valid_logprob + logprob(predictions, b[1]);
                        print("validation set preplexity: %.2f"%float(
                            np.exp(valid_logprob/valid_size)
                        ));