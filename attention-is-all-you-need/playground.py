import torch;
import torch.nn as nn;
import torch.autograd as autograd
import torch.nn.functional as F;
import torch.optim as optim;
import numpy as np;

import keras.backend as keback;
import argparse;
from preprocess import read_instances_from_file;
from preprocess import build_vocab_idx;
import bcolz;
import pickle;


# m1 = np.array([
#     [1,2,3],
#     [1,2,3]
# ],dtype = np.float32)

# m2 = np.array([
#     [1,2],
#     [2,3],
#     [3,4]
# ],dtype = np.float32)

# tensor_m1 = torch.tensor(m1);
# tensor_m2 = torch.tensor(m2);
# print(tensor_m1);
# print(tensor_m2);

# print(tensor_m2.mm(tensor_m1));

# k_tensor_m1 = keback.variable(value = m1);
# k_tensor_m2 = keback.variable(value = m2);

# k_mat_mul = keback.dot(k_tensor_m1, k_tensor_m1);
# print(keback.eval(k_mat_mul));

# linear_model_1 = nn.Linear(2,4);
# linear_input = torch.tensor([
#     [3,4],
#     [7,8],
#     [5,6]
# ], dtype = torch.float32)
# linear_input_transpose = linear_input.transpose(1,0);
# print(linear_input_transpose);
# print('input tensor\n', linear_input);
# output = linear_model_1(linear_input_transpose);
# print(output);

glove_path = '/home/peng/Workspace/data/embeddings/glove.42B.300d.txt';
short_glove_path = '/home/peng/Workspace/data/embeddings/short.glove.42B.300d.txt';
bcolz_embedding_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/bcolz_embeddings.dat'
bcolz_short_embedding_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/short_bcolz_embeddings.dat'
word2idx_path = '/home/peng/Workspace/data/embeddings/word2idx.pkl';


def test_masked_fill():
    a = torch.tensor([1,2,3,5], dtype = torch.float32);
    mask_tensor = torch.ByteTensor([1,1,0,0]);
    b = a.masked_fill(mask_tensor, value = 0);
    print(b);

def test_read_file():
    file_path = '/home/peng/Workspace/data/multi30k/train.en';
    with open(file_path) as f:
        for a in f:
            print(a);

def test_arr_addition():
    a = [[1,2,3]];
    b = [[4,5], [7,8]];
    print(a+b);

def test_arr_indexing():
    a = [1,2,3,4,5];
    print(a[:10]);

def test_read_instances_from_file():
    inst_file = '/home/peng/Workspace/data/multi30k/train.en';
    max_sent_len = 40;
    keep_case = True;
    words_inst = read_instances_from_file(inst_file, max_sent_len, keep_case);
    for sent in words_inst:
        print(sent);

def test_python_extract_zip():
    list1 = ['a','b','c','d'];
    list2 = [1,2,3,4]
    for element in zip(list1, list2):
        print(element);

    zipped_list = zip(list1, list2);
    unzipped_list1, unzipped_list2 = zip(*zipped_list);
    print(unzipped_list1);
    print(unzipped_list2);

def test_nested_for_loop():
    for_set = ['set2_a','set2_b','set2_c'];
    for_list = [w for word in for_set for w in word];
    print(for_list);

def test_build_vocab_idx():
    inst_file = '/home/peng/Workspace/data/multi30k/train.en';
    min_word_count = 2;
    words_inst = read_instances_from_file(inst_file, 30, False);
    words_inst = [w for w in words_inst if w];
    word2idx = build_vocab_idx(words_inst, min_word_count);
    print(word2idx);

def test_word_embedding():
    torch.manual_seed(2);
    word_to_index = {'hello': 0, 'world': 1};
    embeds = nn.Embedding(2,5); # 2 words in vocab, 5 dimensional embeddings.
    lookup_tensor = torch.tensor([word_to_index['hello']], dtype=torch.long);
    hello_embed = embeds(lookup_tensor);
    print(hello_embed);


def test_glove_embeddings():
    words = [];
    idx = 0;
    word2idx = {};
    vectors = [];
    with open(glove_path, 'rb') as f:
        for l in f:
            line_splits = l.decode().split();
            word = line_splits[0];
            words.append(word);
            word2idx[word] = idx;
            idx+=1;
            vect = np.array(line_splits[1:]).astype(np.float);
            vectors.append(vect)
    print(vectors[10]);
    print(word2idx['house']);


def test_write_glove_embeddings_bcolz():
    word2idx = {};
    words = [];
    idx = 0;
    vectors = [];
    with open(glove_path, 'rb') as f:
        for l in f:
            line_splits = l.decode().split();
            word = line_splits[0];
            words.append(word);
            word2idx[word] = idx;
            idx+=1;
            vect = np.array(line_splits[1:]).astype(np.float);
            vectors.append(vect)
    vectors = np.reshape(vectors, newshape=[-1,300]);
    vectors = bcolz.carray(vectors, rootdir= f'{bcolz_embedding_path}', mode='w');
    vectors.flush();
    #vectors = bcolz.open(f'{bcolz_embeddings_path}')[:]
    pickle.dump(word2idx, open(f'{word2idx_path}','wb'))
    #print(vectors[word2idx['house']]);

def test_write_short_glove_embedding_bcolz():
    word2idx = {};
    idx = 0;
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{bcolz_short_embedding_path}', mode='w');
    with open(short_glove_path, 'rb') as f:
        for l in f:
            line_splits = l.decode().split();
            word = line_splits[0];
            word2idx[word] = idx;
            idx+=1;
            vect = np.array(line_splits[1:]).astype(np.float);
            vectors.append(vect);
    vectors = np.reshape(vectors[1:], newshape=[-1, 300]);
    print(vectors.shape);
    #vectors = bcolz.carray();


def test_load_bcolz_embeddings():
    bcolz_embeddings_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/bcolz_embeddings.dat';
    word2idx_path = '/home/peng/Workspace/data/embeddings/word2idx.pkl';
    vectors = bcolz.open(f'{bcolz_embeddings_path}', mode='r');
    word2idx = pickle.load(open(word2idx_path, 'rb'));
    print(vectors[word2idx['house']]);

# extract the first X rows of the embedding, to play with
def extract_partial_embedding():
    head_lines = 20;
    idx = 0;
    with open(glove_path, 'r') as f:
        with open(short_glove_path, 'w') as w:
            for line in f:
                print(line);
                if '\n' in line:
                    print('line contains linebreak.');
                idx+=1;
                if(idx == head_lines):
                    break;
                else:
                    w.write(line);
            w.close();
        f.close();
    print('short version of embedding finished.');


def test_load_embedding():
    # Read in vectors
    vectors = bcolz.open(bcolz_embedding_path, mode='r');
    print(vectors.shape);
    # Read in word2idx
    word2idx = pickle.load(open(word2idx_path, 'rb'));
    print(word2idx['kid']);
    print("kid's embedding is: ");
    print(vectors[word2idx['kid']]);

def test_lstm():
    batch_size = 10;
    seq_len = 3;
    embedding_size = 20;
    num_layer = 2;
    hidden_size = 20;
    lstm_model = nn.LSTM(embedding_size,embedding_size);

    inputs = [
        [torch.randn(1,embedding_size) for _ in range(batch_size)]
        for _ in range(seq_len)
    ]

    inputs = [autograd.Variable(torch.randn(batch_size, embedding_size))]

    hidden = autograd.Variable(torch.randn(num_layer, batch_size, hidden_size))
    #print(hidden);

    for input in inputs:
        out,hidden = lstm_model(input, hidden);
        print(hidden);
    




#test_masked_fill();
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.parse_args();
    #test_read_file();
    #test_arr_addition()
    #test_arr_indexing();
    #test_read_instances_from_file();
    #test_python_extract_zip();
    #test_nested_for_loop();
    #test_build_vocab_idx();
    #test_word_embedding();
    #test_glove_embeddings();
    #test_write_glove_embeddings_bcolz();
    #test_load_bcolz_embeddings();
    #extract_partial_embedding();
    #test_write_glove_embedding_bcolz();
    #test_load_embedding();
    test_lstm();
