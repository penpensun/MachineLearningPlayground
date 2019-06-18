import torch;
import torch.nn as nn;
from model import TestLstm;
import numpy as np;
import datetime;
import configuration as config;
import embeddings as embeds;
import preprocess;
import batch_gen;

'''
This function performs the training;
'''
def train():
    print('training started ...');
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor'); # set all variables to cuda

    print('start reading data ...');
    train_data = preprocess.run_pipeline(); # run the preprocessing pipeline
    print('start splitting train dataset ...');
    x_train, y_train, x_val, y_val = batch_gen.split_dataset(); # split the train data set

    model = TestLstm(feature_size = config.feature_size, hidden_size = config.hidden_size).cuda(); # Define the model
    loss = nn.CrossEntropyLoss(); # Define the loss function
    optimizer = torch.optim.Adam(model.parameters()); # Define the optimizer

    max_batch_idx = len(config.x_train) // config.batch_size; # get the maximum batch index
    
    train_losses = [];
    val_losses = [];
    train_accuracies = [];
    val_accuracies = [];

    for epoch_idx in range(config.epoch_size):
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
            batch_x = batch_gen.get_batch(idx, x_train); # get the batch features
            batch_y = batch_gen.get_batch(idx, y_train); # get the batch targets

            print(batch_x);
            
            
            # # train in the batch
            # for in_batch_idx in range(len(batch_x)):
            #     # get the embeddings
            #     comment_text = batch_x.iloc[in_batch_idx,:].comment_text_adjusted; # get the comment string
            #     if not comment_text.strip():
            #         continue;
            #     target_score = torch.tensor(batch_y.iloc[in_batch_idx], dtype = torch.long); # get the target score
            #     comment_splits = comment_text.split(); # split the comment text
            #     comment_text_idx = [word2idx.get(comment_splits[j], word2idx['unk']) for j in range(len(comment_splits)) ]; # get the word2idx indexes.
            #     #print('comment text idx: ', comment_text_idx);
            #     comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
            #     comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
            #     out = model(comment_text_embeds.view([len(comment_splits), 1, pp.embedding_size])); #compute the cost
            #     # target_score = target_score.long();
            #     # compute the loss
            #     loss_value = loss(out.view(-1, 2), target_score.view(-1));
            #     loss_value.backward(); # compute the backward gradient
            #     optimizer.step(); # carry on the training
            #     train_loss += loss_value; # add up train loss
            #     train_accuracy += 1 if torch.argmax(out) == target_score.data else 0;
                # if torch.argmax(out) != target_score.data:
                #     print('out value: ', out);
                #     print('argmax out value: ', torch.argmax(out));
                #     print('target score: ', target_score.data);
                #     print('train accuracy: ', train_accuracy);
               
        
        # after one epoch, test the model on val
        # for val_idx in range(len(x_val)):
        #     comment_text = x_val.iloc[val_idx,:].comment_text_adjusted; # get the comment string
        #     if not comment_text.strip():
        #         continue;
        #     target_score = torch.tensor(y_val.iloc[val_idx], dtype = torch.long); # get the target score
        #     comment_splits = comment_text.split(); # split the comment text
        #     comment_text_idx = [word2idx.get(comment_splits[j], word2idx['unk']) for j in range(len(comment_splits)) ]; # get the word2idx indexes.
        #     comment_text_embeds = [embeddings[comment_text_idx[j]] for j in range(len(comment_text_idx))] #get the embeddings
        #     comment_text_embeds = torch.Tensor(np.array(comment_text_embeds).astype(np.float16)); # convert the data type.
        #     out = model(comment_text_embeds.view([len(comment_splits),1,pp.embedding_size])); #compute the cost
        #     #target_score = target_score.float();

        #     loss_value = loss(out.view(-1, 2), target_score.view(-1));
        #     val_loss += loss_value;
        #     val_accuracy += 1 if torch.argmax(out) == target_score.data else 0;
        
        # # Append the loss to the train_losses and val_losses lists
        # train_loss = train_loss.item() / len(x_train);
        # val_loss = val_loss.item() / len(x_val);
        # #print('train_loss: ', train_loss);
        # train_losses.append(train_loss);
        # val_losses.append(val_loss);
        
        # # Append the accuracy to train_accuracies and val_accuracies lists
        # train_accuracy = train_accuracy / len(x_train);
        # val_accuracy = val_accuracy / len(x_val);
        # train_accuracies.append(train_accuracy);
        # val_accuracies.append(val_accuracy);
        
        # endtime = datetime.datetime.now(); #get end time
        # print('end time - start time: ', endtime - starttime);
        # # output the results
        # # loss on train data:
        # print('epoch ',epoch_idx+1);
        # #print('training loss: ', train_loss);
        # #print('val loss: ', val_loss);
        # print('train accuracy: ', train_accuracy);
        # print('val accuracy: ', val_accuracy);
    
    # output the train loss and val_loss to file
    # pd.Series(train_losses).to_csv(train_loss_file);
    # print('train_loss dump finished.');
    # #pickle.dump(val_losses, open(val_loss_file, 'w'));
    # pd.Series(val_losses).to_csv(val_loss_file);
    # print('val_loss dump finished.');
    # #pickle.dump(train_accuracy, open(train_loss_file,'w'));
    # pd.Series(train_accuracies).to_csv(train_accuracy_file);
    # print('train accuracy dump finished.');
    # pd.Series(val_accuracies).to_csv(val_accuracy_file);
    # print('val accuracy dump finished.');

    # save model
    #torch.save (model.state_dict(), pp.model_file);
    #print('model saved.');


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
    train();


if __name__ == '__main__':
    run();
    #test_result();