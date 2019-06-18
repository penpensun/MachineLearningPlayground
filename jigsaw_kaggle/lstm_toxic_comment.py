import pandas as pd;
import numpy as np;
import torch.nn as nn;
import torch;
import torch.autograd as autograd;
import bcolz;
import pickle;
import preprocessing as pp;
import random;
import re;
from model import TestLstm;
import preprocess;
import batch_gen;
import configuration as config;

def test_testlstm_with_softmax():
    
    # create the TestLstm class
    test_lstm_model = TestLstm()
    # define the loss function
    loss_func = torch.nn.CrossEntropyLoss();
    # define the optimizer 
    optimizer = torch.optim.Adam(test_lstm_model.parameters());

    # get the train data
    train_data = preprocess.run_pipeline();
    
    # split the train data
    x_train, y_train, x_val, y_val = batch_gen.split_dataset();

    for i in range(len(train_data)):
        comment_text = train_data.iloc[i,:].comment_text_adjusted; # get the comment string
        #target_score = torch.tensor(train_data.iloc[i,:].target, dtype = torch.float16); # get the target score
        target_score = torch.tensor(train_data.iloc[i, :].is_toxic, dtype = torch.float16); # get the target score
        comment_text_idx = [word2idx.get(comment_text[j], word2idx['unk']) for j in range(len(comment_text)) ]; # get the word2idx indexes.
        comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
        comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
        
        out = test_lstm_model(comment_text_embeds.view([seq_num,1,embedding_size])); #compute the cost
        #print(train_data.iloc[i,:]);
        #print('size of out: ',out.size());
        #print('output shape: ');
        #print(out.shape);
        #print('target score: ');
        #print(target_score.view(-1).size());
        target_score = target_score.long();
        # compute the loss
        loss = loss_func(out.view(-1,2), target_score.view(-1));
        #print('loss is: ', loss);
        # compute the backward gradient
        loss.backward();
        # optimize the weights
        optimizer.step();
        #print(out);

if __name__ == '__main__':
    #create_short_train_file();
    #test_testlstm();
    #test_get_batch();
    #preprocessing();
    #create_short_train_file();
    #train();
    #check_default_value_dic();
    #test_testlstm_with_softmax();
    
