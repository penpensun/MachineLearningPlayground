import pandas as pd;
import numpy as np;
import torch.nn as nn;
import torch;
import torch.autograd as autograd;
import bcolz;
import pickle;

# The path of glove embeddings
toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/train.csv';
short_toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/short_train.csv';
bcolz_embedding_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/bcolz_embeddings.dat';
word2idx_path = '/home/peng/Workspace/data/embeddings/word2idx.pkl';
batch = 100;
hidden_num = 100;
seq_num = 300;
embedding_size = 300;


class TestLstm(nn.Module):
    def __init__(self):
        super(TestLstm, self).__init__(); #call the __init__ from super
        self.lstm_layer = nn.LSTM(seq_num, hidden_num)
        self.linear_layer = nn.Linear(hidden_num, 1);
        #self.linear_layer = nn.Linear(hidden_num, 6);
        #self.softmax_layer = nn.Softmax();

    def forward(self, inputs):
        out, (hidden, cell_state) = self.lstm_layer(inputs);
        #out = self.linear_layer(out);
        #out = self.softmax_layer(out);
        return out, (hidden, cell_state);


def create_short_train_file ():
    with open (toxic_comment_input_path, 'r') as f_train:
        with open(short_toxic_comment_input_path, 'w') as f_short_train:
            idx = 0;
            max_idx = 500;
            for line in f_train:
                f_short_train.write(line);
                idx += 1;
                if idx == 100:
                    break;

def test_testlstm():
    # Read in the data
    train_data = pd.read_csv(short_toxic_comment_input_path);
    # Read the embedding
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r');
    # Read in the word2idx
    word2idx = pickle.load(open(word2idx_path, 'rb'));
    # create the TestLstm class
    test_lstm_model = TestLstm()
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
        # get the comment string
        comment_text = train_data.iloc[i,:].comment_text_adjusted;
        # get the target score
        target_score = train_data.iloc[i,:].target;
        # get the embedding of the comment_text
        # get the word2idx indexes.
        comment_text_idx = [word2idx.get(comment_text[j], word2idx['unk']) for j in range(len(comment_text)) ];
        # get the embeddings.
        comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))]
        
        #print(type(comment_text_embeds));
        #print(len(comment_text_embeds));
        comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16));
        #print(comment_text_embeds.shape);
        # create a test tensor
        #test_tensor = torch.randn([300,1,300])
        out = test_lstm_model(comment_text_embeds.view([seq_num,1,embedding_size]));
        print(out);
        # for test, break at loop2
        if i == 0: 
            break;

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
    test_testlstm();
    #check_default_value_dic();
    
