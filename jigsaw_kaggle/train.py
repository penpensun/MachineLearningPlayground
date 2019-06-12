import preprocessing as pp;
import torch;
import torch.nn as nn;
from model import TestLstm;
import numpy as np;
import datetime;


word2idx, embeddings = pp.load_word2idx_embeddings()
print(len(word2idx));
print(len(embeddings));

import pandas as pd;
data = pd.read_csv(pp.short_toxic_comment_input_path)
print(len(data));

# perform preprocessing on comments
data = pp.comment_preprocessing(data);
target_col_name = 'is_toxic';

# for i in range(len(data)):
#     if (i == 1):
#         break;
#     comment_text = data.iloc[i, :].comment_text_adjusted;
#     comment_text_splits = comment_text.split();
#     for j in range(len(comment_text_splits)):
#         print(comment_text_splits[j])

        #print(word2idx[comment_text_splits[j]])

# random select sample
# toxic_index, non_toxic_index = ltc.random_sample_select(data, toxic_num = 5000, non_toxic_num = 5000, target_col_name = target_col_name);
# selected_index = toxic_index + non_toxic_index
# # output the sample into short_toxic_comment file
# data.iloc[selected_index, :].to_csv(ltc.short_toxic_comment_input_path);
# print('selected samples written in file.');

# # load the short toxic comment file
# train_data = pd.read_csv(ltc.short_toxic_comment_input_path);
# print('train data loading finished.');

# # split the data into x_train, y_train, x_val and y_val
# x_train, y_train, x_val, y_val = ltc.dataset_split(train_data, target_col_name);

'''
This function performs the training;
'''
def train(**kwargs):
    print('training started ...');
    
    word2idx = kwargs['word2idx']; # get word2idx 
    embeddings = kwargs['embeddings']; # get embeddigns
    x_train = kwargs['x_train'];
    x_val = kwargs['x_val'];
    y_train = kwargs['y_train'];
    y_val = kwargs['y_val'];
    batch_size = kwargs['batch_size'];
    train_loss_file = kwargs['train_loss_file'];
    val_loss_file = kwargs['val_loss_file'];
    train_accuracy_file = kwargs['train_accuracy_file'];
    val_accuracy_file = kwargs['val_accuracy_file'];

    torch.set_default_tensor_type('torch.cuda.FloatTensor');

    model = TestLstm(feature_size = pp.feature_size, hidden_size = pp.hidden_size).cuda(); # Define the model
    loss = nn.CrossEntropyLoss(); # Define the loss function
    optimizer = torch.optim.Adam(model.parameters()); # Define the optimizer

    max_batch_idx = len(x_train) // batch_size; # get the maximum batch index
    
    train_losses = [];
    val_losses = [];
    train_accuracies = [];
    val_accuracies = [];

    for epoch_idx in range(pp.epoch_size):
        print("epoch: ", (epoch_idx+1))
        starttime = datetime.datetime.now(); # get start time
        # for test
        #if epoch_idx > 0:
         #   break;
        train_loss = 0;
        val_loss = 0;
        train_accuracy = 0;
        val_accuracy = 0;
        # for each batch train the data 
        for idx in range(max_batch_idx+1):
            batch_x = pp.get_batch(idx, x_train); # get the batch features
            batch_y = pp.get_batch(idx, y_train); # get the batch targets

            # train in the batch
            for in_batch_idx in range(len(batch_x)):
                # get the embeddings
                comment_text = batch_x.iloc[in_batch_idx,:].comment_text_adjusted; # get the comment string
                if not comment_text.strip():
                    continue;
                target_score = torch.tensor(batch_y.iloc[in_batch_idx], dtype = torch.long); # get the target score
                comment_splits = comment_text.split(); # split the comment text
                comment_text_idx = [word2idx.get(comment_splits[j], word2idx['unk']) for j in range(len(comment_splits)) ]; # get the word2idx indexes.
                #print('comment text idx: ', comment_text_idx);
                comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
                comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
                out = model(comment_text_embeds.view([len(comment_splits), 1, pp.embedding_size])); #compute the cost
                # target_score = target_score.long();
                # compute the loss
                loss_value = loss(out.view(-1, 2), target_score.view(-1));
                loss_value.backward(); # compute the backward gradient
                optimizer.step(); # carry on the training
                train_loss += loss_value; # add up train loss
                train_accuracy += 1 if torch.argmax(out) == target_score.data else 0;
                # if torch.argmax(out) != target_score.data:
                #     print('out value: ', out);
                #     print('argmax out value: ', torch.argmax(out));
                #     print('target score: ', target_score.data);
                #     print('train accuracy: ', train_accuracy);
               
        
        # after one epoch, test the model on val
        for val_idx in range(len(x_val)):
            comment_text = x_val.iloc[val_idx,:].comment_text_adjusted; # get the comment string
            if not comment_text.strip():
                continue;
            target_score = torch.tensor(y_val.iloc[val_idx], dtype = torch.long); # get the target score
            comment_splits = comment_text.split(); # split the comment text
            comment_text_idx = [word2idx.get(comment_splits[j], word2idx['unk']) for j in range(len(comment_splits)) ]; # get the word2idx indexes.
            comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
            comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
            out = model(comment_text_embeds.view([len(comment_splits),1,pp.embedding_size])); #compute the cost
            #target_score = target_score.float();

            loss_value = loss(out.view(-1, 2), target_score.view(-1));
            val_loss += loss_value;
            val_accuracy += 1 if torch.argmax(out) == target_score.data else 0;
        
        # Append the loss to the train_losses and val_losses lists
        train_loss = train_loss.item() / len(x_train);
        val_loss = val_loss.item() / len(x_val);
        #print('train_loss: ', train_loss);
        train_losses.append(train_loss);
        val_losses.append(val_loss);
        
        # Append the accuracy to train_accuracies and val_accuracies lists
        train_accuracy = train_accuracy / len(x_train);
        val_accuracy = val_accuracy / len(x_val);
        train_accuracies.append(train_accuracy);
        val_accuracies.append(val_accuracy);
        
        endtime = datetime.datetime.now(); #get end time
        print('end time - start time: ', endtime - starttime);
        # output the results
        # loss on train data:
        print('epoch ',epoch_idx+1);
        #print('training loss: ', train_loss);
        #print('val loss: ', val_loss);
        print('train accuracy: ', train_accuracy);
        print('val accuracy: ', val_accuracy);
    
    # output the train loss and val_loss to file
    pd.Series(train_losses).to_csv(train_loss_file);
    print('train_loss dump finished.');
    #pickle.dump(val_losses, open(val_loss_file, 'w'));
    pd.Series(val_losses).to_csv(val_loss_file);
    print('val_loss dump finished.');
    #pickle.dump(train_accuracy, open(train_loss_file,'w'));
    pd.Series(train_accuracies).to_csv(train_accuracy_file);
    print('train accuracy dump finished.');
    pd.Series(val_accuracies).to_csv(val_accuracy_file);
    print('val accuracy dump finished.');

    # save model
    torch.save (model.state_dict(), pp.model_file);
    print('model saved.');


# '''
# Got always 0.94 accuracy for train and for val. must check what happens
# '''
# def test_result():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor');
#     data = pp.comment_preprocessing(pd.read_csv(pp.short_toxic_comment_input_path));
#     print('length of data: ', len(data));
#     print('length of toxic data: ', len(data[data['is_toxic'] == 1]));

#     word2idx,embeddings = pp.load_word2idx_embeddings();

#     model = TestLstm(feature_size = pp.feature_size, hidden_size = pp.hidden_size).cuda(); # Define the model
#     loss = nn.CrossEntropyLoss(); # Define the loss function
#     optimizer = torch.optim.Adam(model.parameters()); # Define the optimizer
#     out_array = [];
#     #return;
#     for i in range(len(data)):
#         comment_text = data.iloc[i,:].comment_text_adjusted; # get the comment string
#         if not comment_text.strip():
#             continue;
#         target_score = torch.tensor(data.iloc[i].is_toxic, dtype = torch.long); # get the target score
#         #print('target : ', target_score.item());
#         comment_splits = comment_text.split(); # split the comment text
#         comment_text_idx = [word2idx.get(comment_splits[j], word2idx['unk']) for j in range(len(comment_splits)) ]; # get the word2idx indexes.
#         #print('comment text idx: ', comment_text_idx);
#         comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
#         comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
#         out = model(comment_text_embeds.view([len(comment_splits), 1, pp.embedding_size])); #compute the cost 
#         loss_value = loss(out.view(-1, 2), target_score.view(-1));
#         loss_value.backward(); # compute the backward gradient
#         optimizer.step(); # carry on the training
#         print('out value: ', out, ' argmax: ',torch.argmax(out).item());
#         #out_array.append([out, torch.argmax(out).item()])




def run():
    word2idx, embeddings = pp.load_word2idx_embeddings(); # load word2idx and embeddings
    data = pd.read_csv(pp.short_toxic_comment_input_path); # read in data set
    data = pp.comment_preprocessing(data); # perform comment preprocessing
    x_train, y_train, x_val, y_val = pp.split_dataset(data, 'is_toxic'); # split dataset
    train(
        word2idx = word2idx,
        embeddings = embeddings,
        x_train = x_train,
        x_val = x_val,
        y_train = y_train,
        y_val = y_val,
        batch_size = pp.batch_size,
        train_loss_file = pp.train_loss_file,
        train_accuracy_file = pp.train_accuracy_file,
        val_loss_file = pp.val_loss_file,
        val_accuracy_file = pp.val_accuracy_file
    );


if __name__ == '__main__':
    run();
    #test_result();