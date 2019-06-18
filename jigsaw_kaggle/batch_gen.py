import pandas as pd;
from sklearn.model_selection import train_test_split;
import configuration as config;

'''
This method the batch idx
'''
def get_batch (batch_idx, train_data):
    #print(len(train_data));
    if (batch_idx + 1) * config.batch_size >= len(train_data):
        return train_data[batch_idx * config.batch_size: len(train_data)];
    else:
        #print("second option: ", (batch_idx + 1) * batch_size);
        return train_data[batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size];


'''
This method performs the preprocessing.
'''
def split_dataset(train_data, target_col_name):
    #train_data = pd.read_csv(toxic_comment_input_path); # read the toxic comment data
    # for test purpose, just read in the short version of the input
    #train_data = pd.read_csv(short_toxic_comment_input_path, engine = 'python');
    
    # perform the comment preprocessing
    #train_data = comment_preprocessing(train_data);

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
Test if get batch works
'''
def test_get_batch():
    # Read the train data
    train_data = pd.read_csv(config.toxic_comment_input_path);
    # test get batch
    batch_dataset = get_batch(0, train_data);
    #print(train_data.shape);
    print(batch_dataset);
    print(batch_dataset.shape);
        