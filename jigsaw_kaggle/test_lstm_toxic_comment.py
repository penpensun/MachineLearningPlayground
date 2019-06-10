from lstm_toxic_comment import toxic_comment_input_path, short_toxic_comment_input_path,\
        bcolz_embedding_path, word2idx_path, batch_size, hidden_num, seq_num,\
        embedding_size, target_col_name, epoch_size, result_file, TestLstm;
import pandas as pd;
import numpy as np;
import bcolz;
import pickle;
import torch;


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
    loss_func = torch.nn.CrossEntropyLoss();
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
    
    # if target > 0.5 than the comment is toxic.
    train_data['is_toxic'] = train_data.apply(lambda x:1 if x['target'] > 0.5 else 0, axis = 1);
    
    #print(train_data['comment_text_adjusted'].iloc[0]);

    #print(train_data['comment_text_adjusted'].map(lambda x: len(x)))
    for i in range(len(train_data)):
        comment_text = train_data.iloc[i,:].comment_text_adjusted; # get the comment string
        #target_score = torch.tensor(train_data.iloc[i,:].target, dtype = torch.float16); # get the target score
        target_score = torch.tensor(train_data.iloc[i, :].is_toxic, dtype = torch.float16); # get the target score
        comment_text_idx = [word2idx.get(comment_text[j], word2idx['unk']) for j in range(len(comment_text)) ]; # get the word2idx indexes.
        comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
        comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
        
        out = test_lstm_model(comment_text_embeds.view([seq_num,1,embedding_size])); #compute the cost
        #print(train_data.iloc[i,:]);
        print('out: ',out);
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

def test_dataset_insight():
        dataset = pd.read_csv(short_toxic_comment_input_path, engine = 'python');
        dataset['is_toxic'] = dataset.apply(lambda x: 1 if x['target'] > 0.5 else 0, axis = 1);
        print(dataset[['target','is_toxic']]);

if __name__ == '__main__':
        #test_dataset_insight();
        #test_testlstm();
        ts = torch.tensor(0.1);
        print(ts.item());
