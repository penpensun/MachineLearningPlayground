from __future__ import print_function;
import collections;
import math;
import numpy as np;
import os;
import tensorflow as tf;
import zipfile
from matplotlib import pylab;
from six.moves import range;
from six.moves.urllib.request import urlretrieve;
import six.moves.urllib as urllib;
from sklearn.manifold import TSNE;
import sys;
import random;

url = "http://mattmahoney.net/dc/";

windows_dir = "c:/workspace/data/";
mac_dir = "/Users/penpen926/workspace/data/"

def maybe_download(filename, expected_bytes):
    '''
    Download a file if not present, and make sure it's the right size.
    '''
    if 'win' in sys.platform:
        fullfilename = windows_dir+filename;
    if 'mac' in sys.platform:
        fullfilename = mac_dir +filename;

    if 'win' in sys.platform:
        '''Use proxy'''
        proxy = urllib.request.ProxyHandler({'http':'http://MZMBI:FMRs5Tab@10.185.190.100:8080'});
        opener = urllib.request.build_opener(proxy);
        urllib.request.install_opener(opener);

    if not os.path.exists(filename):
        print("print out the url.");
        print(url+filename);
        filename, _ = urlretrieve(url+filename, fullfilename);
    statinfo = os.stat(filename);
    if statinfo.st_size == expected_bytes:
        print("Found and verified %s" % filename);
    else:
        print(statinfo.st_size);
        raise Exception('Fail to verify'+ filename+'. Can you get it with a browser');
    return filename;

def read_data(filename):
    '''Extract the first file enclosed in a zip file as a list of words'''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split();
    return data;

vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]];
    count.extend(collections.Counter(words).most_common(vocabulary_size-1));
    dictionary = dict();
    for word, _ in count:
        dictionary[word] = len(dictionary);
    data = list();
    unk_count = 0;
    for word in words:
        if word in dictionary:
            index = dictionary[word];
        else:
            index =0; #dictionary['UNK']
            unk_count = unk_count+1;
        data.append(index);
    count[0][1] = unk_count;
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()));
    return data, count, dictionary, reverse_dictionary;



def generate_batch(batch_size,num_skips, skip_window, data):
    global data_index;
    assert batch_size % num_skips == 0;
    assert num_skips <= 2* skip_window;
    batch = np.ndarray(shape = (batch_size),dtype = np.int32);
    labels = np.ndarray(shape = (batch_size,1),dtype = np.int32);

    span = 2*skip_window+1;# [skip_window target skip_window]
    buffer = collections.deque(maxlen = span);

    for _ in range(span):
        buffer.append(data[data_index]);
        data_index = (data_index+1)% len(data);

    for i in range(batch_size // num_skips):
        target = skip_window;
        targets_to_avoid = [skip_window];
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1);
            targets_to_avoid.append(target);
            batch[i*num_skips+j] = buffer[skip_window];
            labels[i*num_skips+j,0] = buffer[target];
        buffer.append(data[data_index]);
        data_index = (data_index+1)%len(data);
    return batch, labels;


filename = maybe_download('text8.zip', 31344016);
data_index = 0;
words = read_data(filename);
print('Data size %d' %len(words));
data,count, dictionary, reverse_dictionary = build_dataset(words);
print('Most common words (+UNK) ',count[:5]);
print('Sample data', data[:10]);

print('data:', [reverse_dictionary[di] for di in data[:8] ]);
for num_skips, skip_window in [(2,1), (4,2)]:
    data_index = 0;
    batch, labels = generate_batch(batch_size = 8, num_skips = num_skips, skip_window = skip_window, data = data);
    print('\n with num_skips = %d and skip_window = %d: '% (num_skips, skip_window));
    print('  batch:', [reverse_dictionary[bi] for bi in batch]);
    print('  labels:', [reverse_dictionary[li] for li in labels.reshape(8)]);

#Print out the data
#print("Print out the data:");
#print(data);

batch_size = 128;
embedding_size = 128; #dimension of the embedding vector.
skip_window = 1; #how many words to consider left and right
num_skips = 2; # how many times to reuse an input to generate a label

valid_size = 16 #Random set of words to evaluate similarity on.
valid_window = 100 #Only pick dev samples in the head of the distribution
valid_examples = np.array(random.sample(range(valid_window), valid_size));
num_sampled = 64; # Number of negative examples to sample


def training():
    graph = tf.Graph();
    with graph.as_default():

        #Input data
        train_dataset = tf.placeholder(tf.int32, shape = [batch_size]);
        train_labels = tf.placeholder(tf.int32, shape= [batch_size,1]);
        valid_dataset = tf.constant(valid_examples, dtype = tf.int32);

        #Variables
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0));
        softmax_weights = tf.Variable(tf.truncated_normal(
            [vocabulary_size, embedding_size],stddev = 1.0/math.sqrt(embedding_size)));

        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]));

        #Model.
        #Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset);
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed, train_labels, num_sampled, vocabulary_size)
        );

        #Optimizer
        # The embeddings are also optimized, because the embeddings are also as variables defined.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss);

    #Start training
    num_steps = 10001;
    with tf.Session(graph = graph) as session:
        tf.initialize_all_variables();
        print("Initialized.");
        average_loss = 0;
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window);
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels};
            _, l = session.run([optimizer, loss], feed_dict = feed_dict);


