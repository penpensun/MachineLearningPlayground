import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torch.optim as optim;
import numpy as np;

import keras.backend as keback;
import argparse;
from preprocess import read_instances_from_file;
from preprocess import build_vocab_idx;

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
    word_to_index = {'hello': 0, 'world'};

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
    test_build_vocab_idx();



