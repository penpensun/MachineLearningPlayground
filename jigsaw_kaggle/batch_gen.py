import pandas as pd;
from sklearn.model_selection import train_test_split;
import configuration as config;
import numpy as np;
import bcolz;

'''
in this function the embeddings for each batch are assigned and the batch is returned.
'''
def assign_embeddings_and_get_batch (batch_idx, train_data,
    embed_col_name = None,
    bcolz_embeddings = None):
    #print(len(train_data));
    # get batch_data
    if (batch_idx + 1) * config.batch_size >= len(train_data):
        batch_data = train_data[batch_idx * config.batch_size: len(train_data)];
    else:
        #print("second option: ", (batch_idx + 1) * batch_size);
        batch_data = train_data[batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size];
    
    # assign embeddings to the batch_data
    #embeddings = [ bcolz_embeddings[embed_idx] for embed_array in batch_data[embed_col_name] for embed_idx in embed_array]
    embeddings = None;
    for embed_array in batch_data[embed_col_name]:
        #print('embed_array: ', embed_array);
        embed_array = embed_array.split(',')
        np.append(embeddings, \
            np.array([bcolz_embeddings[int(embed_idx)] for embed_idx in embed_array]).astype(np.float32));
        if embeddings is None:
            embeddings = np.array([bcolz_embeddings[int(embed_idx)] for embed_idx in embed_array]).astype(np.float32).T;
        else:
            embeddings = np.hstack(
                (embeddings,
                np.array([bcolz_embeddings[int(embed_idx)] for embed_idx in embed_array]).astype(np.float32).T));
    print('inside assign embedding, shape of embeddings: ', embeddings.shape);
    embeddings = np.reshape(embeddings, [-1, 300, config.sentence_length])
    return embeddings;

'''
this function retrieves a batch of targets
'''
def get_batch_target (batch_idx, train_target):

    # get batch_data
    if (batch_idx +1) * config.batch_size >= len(train_target):
        return train_target[batch_idx * config.batch_size: len(train_target)];
    else:
        batch_data = train_target[batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size];
'''
This method performs the preprocessing.
'''
def split_dataset(train_data, target_col, feature_cols):
    #train_data = pd.read_csv(toxic_comment_input_path); # read the toxic comment data
    # for test purpose, just read in the short version of the input
    #train_data = pd.read_csv(short_toxic_comment_input_path, engine = 'python');
    
    # perform the comment preprocessing
    #train_data = comment_preprocessing(train_data);
    train_features = train_data[feature_cols] # get the train features
    print('inside split dataset, feature_cols: ', feature_cols);
    print(train_data.columns.values);
    train_target = train_data[target_col]; # get the train targets
    print('shape of train_fea    ', train_features.shape);
    print('shape of train_tar    ', train_target.shape);
    x_train, x_val, y_train, y_val = train_test_split(train_features, train_target, test_size = 0.05, random_state = 10);
    print('shape of x_train: ', x_train.shape);
    print('shape of x_val: ', x_val.shape);
    print('shape of y_train: ', y_train.shape);
    print('shape of y_val: ', y_val.shape);
    return x_train, y_train, x_val, y_val;

'''
Test if get batch works
'''
def test_assign_embeddings_and_get_batch ():
    batch_idx = 0;
    train_data = pd.read_csv(config.processed_toxic_comment_input_path); # read test_train_data
    train_x = train_data[config.feature_cols]; # assign x (feature) data
    train_y = train_data[config.target_col]; # assign y (target) data
    bcolz_embeddings = bcolz.open(config.bcolz_embedding_path, mode = 'r');
    # get index 0 batch
    batch_x = assign_embeddings_and_get_batch(0, train_data, config.feature_cols[0], bcolz_embeddings);
    print('shape of batch_x: ', batch_x.shape);
    print(batch_x);


if __name__ == '__main__':
    test_assign_embeddings_and_get_batch();
        