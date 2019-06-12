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


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__();
        self.linear = nn.Linear(300, 10);
    
    def forward(self, input):
        self.hidden = self.linear(input);
        #self.softmax_values = nn.functional.softmax(self.hidden, dim = 0);
        #return self.softmax_values
        return self.hidden;


def run_test_model():
    # test sequence
    seq = ['this','is','a','test','run'];
    # load embeddings
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r');
    word2idx = pickle.load(open(word2idx_path, 'rb'));

    seq_embeddings = np.array([embeddings[word2idx[seq[i]]] for i in range(len(seq))]).astype(np.float16);
    print(seq_embeddings.shape)
    print(seq_embeddings);

    test_model_1 = TestModel();
    # Check if cuda is available
    print('is cuda available: ', torch.cuda.is_available());
    #seq_embeddings_tensor = torch.cuda.FloatTensor(seq_embeddings);
    seq_embeddings_tensor = torch.Tensor(seq_embeddings);
    print('the device of variable seq_embeddings_tensor: ', seq_embeddings_tensor.device);
    
    # Create cuda device
    cuda_device = torch.device('cuda:0');

    # Transfer the variable to cuda
    seq_embeddings_tensor = seq_embeddings_tensor.to(cuda_device)
    # print the device
    print("the device of seq_embeddings_tensor: ", seq_embeddings_tensor.device);

    # print out the device of the model
    
    print('The device of test_model_1 is cuda: ',next(test_model_1.parameters()).is_cuda);
    
    # transfer the model to cuda
    test_model_1 = test_model_1.to(cuda_device)
    print('The device of test_model_1 is cuda: ', next(test_model_1.parameters()).is_cuda);
    
    # Run the model
    out = test_model_1(seq_embeddings_tensor);
    print(out)
    #print(torch.cuda.is_available());

    # print(out[0,:])
    # print(torch.sum(out[0,:]))
    # # perform softmax
    # print(torch.nn.functional.softmax(out[0,:], dim = 0))
    # print(torch.sum(torch.nn.functional.softmax(out[0,:], dim = 0)));

    # Perform softmax on the output matrix
    softmax_res = torch.nn.functional.softmax(out, dim = 1)
    print('softmax on matrix level:');
    print(softmax_res);
    print('sum of rows:');
    print(torch.sum(softmax_res[0,:]))


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


def gen_embedding():
    word2idx = {};
    embeddings = [];
    
    # open the embedding file
    with open(short_glove_path, 'rb') as f:
        for l in f:
            line_splits = l.decode().split();
            word = line_splits[0];
            embed = np.array(line_splits[1:]).astype(np.float16);
            #print(embed);
            # get the index
            idx = len(embeddings) -1;
            #print('idx: ', idx);
            word2idx[word] = idx;
            # push the embed into the embedding array
            embeddings.append(embed);
    print('test word embeddings: ');
    print(embeddings[word2idx['the']]);


def gen_bcolz_embedding():
    word2idx = {};
    embeddings = []
    with open(glove_path, 'rb') as f:
        for l in f:
            # split the line
            line_splits = l.decode().split();
            # get the word
            word = line_splits[0];
            # get the index
            idx = len(embeddings);
            # assign the value in word2idx
            word2idx[word] = idx;
            # append the embed to embeddings
            embeddings.append(np.array(line_splits[1:]).astype(np.float16));
    
    # write to bcolz. 
    #bcolz_vectors = bcolz.carray(embeddings, rootdir=bcolz_embedding_path, mode = 'w');
    print('bcolz write finished');
    # write index with pickle
    print('dump the word2idx.');
    pickle.dump(word2idx, open(word2idx_path, 'wb'));
    print('read the word2idx');
    test_word2idx = pickle.load(open(word2idx_path, 'rb'));
    print('test word2idx: terrorist');
    print(test_word2idx['terrorist']);



def read_bcolz_embedding():
    # open the bcolz and read out the embeddings
    embeddings = bcolz.open(rootdir=bcolz_embedding_path, mode='r');
    # load the word2idx
    word2idx = pickle.load(open(word2idx_path, 'rb'));
    # test the word embeddings
    print('Embedding for: terrorist');
    print(embeddings[word2idx['terrorist']]);
    


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
    num_layers = 2;
    hidden_size = 100;
    embedding_size = 300;
    batch_size = 50;
    seq_len = 50;
    lstm_model = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers);

    #Create inputs
    inputs = autograd.Variable(torch.randn(size = (seq_len, batch_size, embedding_size)));
    # Run the model
    out, hidden = lstm_model(inputs);
    print('out:');
    print(out);
    print('hidden:');
    print(hidden);

    print('out type:');
    print(type(out));
    print('Size of out:');
    print(out.size());

    print('hidden type: ');
    print(type(hidden));
    print(len(hidden));
    print('size of the hidden result: ');
    print(hidden[0].size());
    print('size of the cell state: ');
    print(hidden[1].size());

    
def test_autograd():
    test_var = autograd.Variable(torch.Tensor([1]), requires_grad = True);
    cuda_test_var = test_var.cuda();
    cpu_test_var = test_var.cpu();
    print('cuda test variable: ', cuda_test_var);
    print('cpu test variable: ', cpu_test_var);

    print('data');
    print('cuda variable data: ', cuda_test_var.data);
    print('cpu variable data: ', cpu_test_var.data);

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
    #test_lstm();
    #test_autograd();
    #gen_embedding();
    #gen_bcolz_embedding();
    #read_bcolz_embedding();
    run_test_model();
