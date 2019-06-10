import pandas as pd;
import bcolz;
import pickle;
import random;
import re;
from sklearn.model_selection import train_test_split;

# The path of glove embeddings
toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/train.csv';
short_toxic_comment_input_path = '/home/peng/Workspace/data/kaggle/jigsaw/short_train.csv';
bcolz_embedding_path = '/home/peng/Workspace/data/embeddings/bcolz_vectors/bcolz_embeddings.dat';
word2idx_path = '/home/peng/Workspace/data/embeddings/word2idx.pkl';
batch_size = 100;
hidden_size = 100;
feature_size = 300;
embedding_size = 300;
target_col_name = 'is_toxic'
epoch_size = 10;
result_file = '/home/peng/Workspace/kaggle/jigsaw/result.csv';
train_loss_file = '/home/peng/Workspace/kaggle/jigsaw/train_loss.txt';
val_loss_file = '/home/peng/Workspace/kaggle/jigsaw/val_loss.txt';
train_accuracy_file = '/home/peng/Workspace/kaggle/jigsaw/train_accuracy_file.txt';
val_accuracy_file = '/home/peng/Workspace/kaggle/jigsaw/val_accuracy_file.txt';
model_file = '/home/peng/Workspace/kaggle/jigsaw/model.txt';

'''
This function create a shorter version of the input data, for test purpse
'''
def create_short_train_file ():
    with open (toxic_comment_input_path, 'r') as f_train:
        with open(short_toxic_comment_input_path, 'w') as f_short_train:
            idx = 0;
            max_idx = 5001;
            for line in f_train:
                f_short_train.write(line);
                idx += 1;
                if idx == max_idx:
                    break;
    print("short train file successfully created.");
'''
This method get the batch idx
'''
def get_batch (batch_idx, train_data):
    #print(len(train_data));
    if (batch_idx + 1) * batch_size >= len(train_data):
        return train_data[batch_idx * batch_size: len(train_data)];
    else:
        #print("second option: ", (batch_idx + 1) * batch_size);
        return train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size];

'''
This method performs the comment preprocessing.
'''
def comment_preprocessing(df):
    print('inside comment preprocessing.');
    # Adjust the length of the comments:
    #df['comment_text_adjusted'] = df.apply(lambda x: x['comment_text'].ljust(seq_num, ' '), axis = 1);
    # Truncate the comment_text.
    #df['comment_text_adjusted'] = df.comment_text_adjusted.map(lambda x: x[:seq_num] if len(x) > seq_num else x);
    # filter out the non-number and non alphabetical characters
    df['comment_text_adjusted'] = df.comment_text.map(filter_special_char);
    # turn all letters to lower case.
    df['comment_text_adjusted'] = df.comment_text_adjusted.map(lambda x: x.lower());
    # if target > 0.5, than the commit is toxic
    df['is_toxic'] = df.apply(lambda x: 1 if x['target'] > 0.5 else 0, axis = 1);
    return df;

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
This function loads word2idx matrix, embeddings, 
'''
def load_word2idx_embeddings():
    print('load word2idx...', end='');
    word2idx = pickle.load(open(word2idx_path,'rb')); # load the word2idx matrix
    print('finished.');
    print('load embeddings...', end='');
    embeddings = bcolz.open(bcolz_embedding_path, mode='r'); #load the embeddings
    print('finished.');
    return word2idx, embeddings;

''' 
This function randomly select X number of non-toxic comments and Y number of toxic comments
'''
def random_sample_select(train_data, toxic_num, non_toxic_num, target_col_name):
    toxic_index = random.sample(train_data[train_data[target_col_name] == 1].index.values.tolist(), toxic_num);
    non_toxic_index = random.sample(train_data[train_data[target_col_name] == 0].index.values.tolist(), non_toxic_num);
    return toxic_index, non_toxic_index;


'''
This function filters off the special characters in the string
'''
def filter_special_char (s):
    return ''.join(c for c in s if c.isalnum() or c == ' ')


def test_filter_special_char():
    s1 = 'good luck! than.,.k you v@!ery much!!!?!?';
    print(filter_special_char(s1))


# Test functions

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
        

def check_default_value_dic():
    word2idx = pickle.load(open(word2idx_path, 'rb'))
    embeddings = bcolz.open(bcolz_embedding_path, mode = 'r');
    print('s index: ');
    #print(word2idx.get('do\'t', word2idx['unk']));
    #print(word2idx['s'])
    # set up the default value of the word2idx dictionary
    #word2idx.setdefault()
