import pandas as pd;
import numpy as np;
import torch.nn as nn;
import torch;
import torch.autograd as autograd;
import bcolz;
import pickle;
from sklearn.model_selection import train_test_split;

# The path of glove embeddings
toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/train.csv';
short_toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/short_train.csv';
bcolz_embedding_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/bcolz_embeddings.dat';
word2idx_path = '/home/peng/Workspace/data/embeddings/word2idx.pkl';
batch_size = 100;
hidden_num = 100;
seq_num = 300;
embedding_size = 300;
target_col_name = 'target'


class TestLstm(nn.Module):
    def __init__(self):
        super(TestLstm, self).__init__(); #call the __init__ from super
        self.lstm_layer = nn.LSTM(seq_num, hidden_num)
        self.linear_layer = nn.Linear(hidden_num, 1);
        #self.linear_layer = nn.Linear(hidden_num, 6);
        #self.softmax_layer = nn.Softmax();

    def forward(self, inputs):
        out, (hidden, cell_state) = self.lstm_layer(inputs);
        out = self.linear_layer(out);
        #out = self.softmax_layer(out);
        print('inside forward, out shape: ', out.size());
        return out[seq_num-1, 0,0];

'''
This function create a shorter version of the input data, for test purpse
'''
def create_short_train_file ():
    with open (toxic_comment_input_path, 'r') as f_train:
        with open(short_toxic_comment_input_path, 'w') as f_short_train:
            idx = 0;
            max_idx = 2000;
            for line in f_train:
                f_short_train.write(line);
                idx += 1;
                if idx == max_idx:
                    break;

'''
This method get the batch idx
'''
def get_batch (batch_idx, train_data):
    print(len(train_data));
    if (batch_idx + 1) * batch_size >= len(train_data):
        return train_data[batch_idx * batch_size: len(train_data)];
    else:
        print("second option: ", (batch_idx + 1) * batch_size);
        return train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size];

'''
This method performs the preprocessing.
'''
def preprocessing():
    #train_data = pd.read_csv(toxic_comment_input_path); # read the toxic comment data
    # for test purpose, just read in the short version of the input
    train_data = pd.read_csv(short_toxic_comment_input_path);
    
    # Adjust the length of the comments:
    train_data['comment_text_adjusted'] = train_data.apply(lambda x: x['comment_text'].ljust(seq_num, ' '), axis = 1);
    # Truncate the comment_text.
    train_data['comment_text_adjusted'] = train_data.comment_text_adjusted.map(lambda x: x[:seq_num] if len(x) > seq_num else x);
    # turn all letters to lower case.
    train_data['comment_text_adjusted'] = train_data.comment_text_adjusted.map(lambda x: x.lower());

    # split the train features and train targets
    columns = train_data.columns.tolist();
    columns.remove(target_col_name)

    train_features = train_data[columns] # get the train features
    train_target = train_data[target_col_name]; # get the train targets
    print('shape of train_feature ', train_features.shape);
    print('shape of train_target ', train_target.shape);
    x_train, x_val, y_train, y_val = train_test_split(train_features, train_target, test_size = 0.05, random_state = 10);

    print('shape of x_train: ', x_train.shape);
    print('shape of x_val: ', x_val.shape);
    print('shape of y_train: ', y_train.shape);
    print('shape of y_val: ', y_val.shape);
    return x_train, y_train, x_val, y_val;

'''
This function performs the training;
'''
def train():
    word2idx = pickle.load(open(word2idx_path,'rb')); # load the word2idx matrix
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r'); # load the embeddings
    
    model = TestLstm(); # Define the model
    loss = nn.MSELoss(); # Define the loss function
    optimizer = torch.optim.Adam(model.parameters()); # Define the optimizer

    x_train, y_train, x_val, y_val = preprocessing(); # perform the preprocessing
    max_batch_idx = len(x_train) // batch_size; # get the maximum batch index
    # for each batch train the data 
    for idx in range(max_batch_idx+1):
        batch_x = get_batch(idx, x_train); # get the batch features
        batch_y = get_batch(idx, y_train); # get the batch targets

        # train in the batch
        for in_batch_idx in range(len(batch_x)):
            # get the embeddings
            comment_text = batch_x.iloc[in_batch_idx,:].comment_text_adjusted; # get the comment string
            target_score = torch.tensor(batch_y.iloc[in_batch_idx], dtype = torch.float16); # get the target score
            comment_text_idx = [word2idx.get(comment_text[j], word2idx['unk']) for j in range(len(comment_text)) ]; # get the word2idx indexes.
            comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
            comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
            out = model(comment_text_embeds.view([seq_num,1,embedding_size])); #compute the cost
            target_score = target_score.float();
            # compute the loss
            loss_value = loss(out, target_score);
            loss_value.backward(); # compute the backward gradient
            optimizer.step(); # carry on the training
            print('loss: ', loss_value);
        


'''
This function splits the dataset into
'''
def split_dataset ():
    pass;

'''
Test if get batch works
'''
def test_get_batch():
    # Read the train data
    train_data = pd.read_csv(toxic_comment_input_path);
    # test get batch
    batch_dataset = get_batch(0, train_data);
    #print(train_data.shape);
    print(batch_dataset);
    print(batch_dataset.shape);
        

def test_testlstm():
    # Read in the data
    train_data = pd.read_csv(short_toxic_comment_input_path);
    # Read the embedding
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r');
    # Read in the word2idx
    word2idx = pickle.load(open(word2idx_path, 'rb'));
    # create the TestLstm class
    test_lstm_model = TestLstm()
    # define the loss function
    loss_func = torch.nn.MSELoss();
    # define the optimizer 
    optimizer = torch.optim.Adam(test_lstm_model.parameters());
    # calculate the mean of the comment text
    #print(np.mean(train_data['comment_text'].map(lambda x: len(x)).values));
    
    # Perform the preprocessing:
    # Adjust the length of the comment text.
    train_data['comment_text_adjusted'] = train_data.apply(lambda x: x['comment_text'].ljust(seq_num, ' '), axis = 1);
    # Truncate the comment_text.
    train_data['comment_text_adjusted'] = train_data.comment_text_adjusted.map(lambda x: x[:seq_num] if len(x) > seq_num else x);
    # turn all letters to lower case.
    train_data['comment_text_adjusted'] = train_data.comment_text_adjusted.map(lambda x: x.lower());
    
    #print(train_data['comment_text_adjusted'].iloc[0]);

    #print(train_data['comment_text_adjusted'].map(lambda x: len(x)))
    for i in range(len(train_data)):
        comment_text = train_data.iloc[i,:].comment_text_adjusted; # get the comment string
        target_score = torch.tensor(train_data.iloc[i,:].target, dtype = torch.float16); # get the target score
        comment_text_idx = [word2idx.get(comment_text[j], word2idx['unk']) for j in range(len(comment_text)) ]; # get the word2idx indexes.
        comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
        comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
        
        out = test_lstm_model(comment_text_embeds.view([seq_num,1,embedding_size])); #compute the cost
        #print(train_data.iloc[i,:]);
        #print('out: ',out);
        #print('output shape: ');
        #print(out.shape);
        #print('target score: ');
        #print(target_score);
        target_score = target_score.float();
        # compute the loss
        loss = loss_func(out, target_score);
        print(loss);
        # compute the backward gradient
        loss.backward();
        # optimize the weights
        optimizer.step();
        #print(out);

def check_default_value_dic():
    word2idx = pickle.load(open(word2idx_path, 'rb'))
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r');
    print('s index: ');
    #print(word2idx.get('do\'t', word2idx['unk']));
    #print(word2idx['s'])
    # set up the default value of the word2idx dictionary
    #word2idx.setdefault()
    

if __name__ == '__main__':
    #create_short_train_file();
    #test_testlstm();
    #test_get_batch();
    #preprocessing();
    #create_short_train_file();
    train();
    #check_default_value_dic();
    
