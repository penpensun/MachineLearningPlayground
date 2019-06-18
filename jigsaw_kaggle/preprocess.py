import pandas as pd;
import bcolz;
import pickle;
import random;
import re;
from sklearn.model_selection import train_test_split;
import configuration as config;
import embeddings as embeds;
import numpy as np;
import math;

'''
This function generates word2idx and uses pickle to output to word2idx_path,
generates embeddings using bcolz to bcolz_embedding_path
'''
def write_glove_embeddings_bcolz():
    word2idx = {};
    #words = [];
    idx = 0;
    vectors = [];
    with open (config.pretrained_embeds_path, 'rb') as f:
        for l in f:
            line_splits = l.decode().split();
            word = line_splits[0];
            #words.append(word);
            word2idx[word] = idx;
            idx +=1;
            vect = np.array(line_splits[1: ]).astype(np.float32);
            vectors.append(vect);
    vectors = np.reshape(vectors, newshape=[-1, 300]);
    vectors = bcolz.carray(vectors, rootdir= f'{config.bcolz_embedding_path}', mode = 'w');
    vectors.flush();
    pickle.dump(word2idx, open(f'{config.word2idx_path}', 'wb'))

'''
This function create a shorter version of the input data, for test purpse
'''
def create_short_train_file ():
    with open (config.toxic_comment_input_path, 'r') as f_train:
        with open(config.short_toxic_comment_input_path, 'w') as f_short_train:
            idx = 0;
            max_idx = 5001;
            for line in f_train:
                f_short_train.write(line);
                idx += 1;
                if idx == max_idx:
                    break;
    print("short train file successfully created.");


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
This function randomly select X number of non-toxic comments and Y number of toxic comments
'''
def random_sample_select(train_data, toxic_num, non_toxic_num, target_col_name):
    toxic_index = random.sample(train_data[train_data[target_col_name] == 1].index.values.tolist(), toxic_num);
    non_toxic_index = random.sample(train_data[train_data[target_col_name] == 0].index.values.tolist(), non_toxic_num);
    return toxic_index, non_toxic_index;

'''
This function reads in the data from file
'''
def read_in_data (debug: bool = False) -> pd.DataFrame:
    # in debug mode, read in short version of the input data
    if not debug:
        data = pd.read_csv(config.toxic_comment_input_path);
    else:
        data  = pd.read_csv(config.short_toxic_comment_input_path);
    return data;
'''
This function handles the missing data
'''
def handle_missing_data (data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    return data;

'''
Adjust comment column
'''
def adjust_comment_col (data: pd.DataFrame ) -> pd.DataFrame:
    # print('inside comment preprocessing.');
    # Adjust the length of the comments:
    #df['comment_text_adjusted'] = df.apply(lambda x: x['comment_text'].ljust(seq_num, ' '), axis = 1);
    # Truncate the comment_text.
    #df['comment_text_adjusted'] = df.comment_text_adjusted.map(lambda x: x[:seq_num] if len(x) > seq_num else x);
    # filter out the non-number and non alphabetical characters
    print('adjust comment text ...')
    data['comment_text_adjusted'] = data.comment_text.map(filter_special_char);
    # turn all letters to lower case.
    data['comment_text_adjusted'] = data.comment_text_adjusted.map(lambda x: x.lower());
    return data;

def gen_is_toxic_col (data: pd.DataFrame) -> pd.DataFrame:
    # if target > 0.5, than the commit is toxic
    print('gen is_toxic column ...');
    data['is_toxic'] = data.apply(lambda x: 1 if x['target'] > 0.5 else 0, axis = 1);
    return data;


'''
This function filters off the special characters in the string
'''
def filter_special_char (s):
    return ''.join(c for c in s if c.isalnum() or c == ' ')


'''
Assign embedding matrix. This function generates a column for the dataframe, assign the embeddings of the comment to the column.
'''
def assign_embeddings(data: pd.DataFrame) -> pd.DataFrame:
    print('assign embeddings ...')
    data['comment_embeddings'] = [[] for _ in range(len(data))];
    data = data.apply(gen_comment_embedding, axis = 1);
    return data;

'''
This function generates the embeddings for a row.
'''
def gen_comment_embedding(data_row: pd.Series) -> pd.DataFrame:
    #print('data row: ',data_row);
    #print('type of data_row: ', type(data_row));
    #print('comment adjusted: ', data_row['comment_text_adjusted']);
    #print('type of comment adjusted: ', type(data_row['comment_text_adjusted']));
    comment_split = data_row['comment_text_adjusted'].split();
    comment_embeddings = np.array(
        [item for item in map(lambda x: embeds.embeddings[embeds.word2idx.get(x, embeds.word2idx['unk'])], comment_split)],
        dtype = np.float16
    );
    data_row['comment_embeddings'] = comment_embeddings;
    return data_row;

'''
This function selects the relevant columns from the dataset and returns the dataset.
'''
def select_data_cols(data: pd.DataFrame) -> pd.DataFrame:
    #print(data);
    return data[
        ['id', 'target','is_toxic','comment_embeddings', 'comment_text_adjusted']
    ]


'''
This function runs the pipeline for preprocessing
'''
def run_pipeline(debug = False):
    train_data = read_in_data(debug); # read in data
    train_data = assign_embeddings( # assign embeddings to the comment column
        gen_is_toxic_col( # generate the is_toxic column
        adjust_comment_col(train_data) # adjust the comment column
        )
        )
    
    return train_data;

'''
This function runs pipeline for preprocessing and 
'''
def run_pipeline_out(debug = False):
    train_data = read_in_data(debug); # read in data
    train_data = assign_embeddings( # assign embeddings to the comment column
        gen_is_toxic_col( # generate the is_toxic column
        adjust_comment_col(train_data) # adjust the comment column
        )
        )
    train_data.write_csv(config.processed_toxic_comment_input_path);

##################################
## Test functions
##################################

def check_default_value_dic():
    word2idx = pickle.load(open(config.word2idx_path, 'rb'))
    embeddings = bcolz.open(config.bcolz_embedding_path, mode = 'r');
    print('s index: ');
    #print(word2idx.get('do\'t', word2idx['unk']));
    #print(word2idx['s'])
    # set up the default value of the word2idx dictionary
    #word2idx.setdefault()

'''
This function tests the pipeline preprocessing
'''
def test_run_pipeline():
    train_data = run_pipeline(True);
    print(train_data);

'''
This function tests assign_embeddings()
'''
def test_assign_embeddings():
    # read in data
    print('toxic file path: ', config.short_toxic_comment_input_path);
    data= read_in_data(debug = True);
    comment_text = 'this is a text sentence';
    df = pd.DataFrame([comment_text], columns =['comment_text_adjusted'] );
    #print(df);
    #print('type of df: ', type(df));
    assign_embeddings(df)
    print('comment text: ', comment_text);
    print('embeddings: ', df['comment_embeddings'].iloc[0])
    embeds_res = df['comment_embeddings'].iloc[0];

    embeds_expected = np.array(
        [item for item in map(lambda x: embeds.embeddings[embeds.word2idx.get(x, 'unk')], 'this is a text sentence'.split())],
        dtype = np.float16
    )

    print(embeds_res == embeds_expected);



def test_filter_special_char():
    s1 = 'good luck! than.,.k you v@!ery much!!!?!?';
    print(filter_special_char(s1))

'''
This function randomly pickup one word embedding and compares it with the bcolz embeddings stored at config.bcolz_embedding_path
'''
def test_embeddings():
    num_lines = 0;
    with open (config.pretrained_embeds_path, 'rb') as fopen:
        for l in fopen:
            num_lines +=1;
        fopen.close();
    print('total num of lines: ', num_lines);
    rand_num_line = np.random.randint(0, num_lines); #randomly select one line
    print('randomly select line: ', rand_num_line);
    idx = 0; # index of the line
    with open(config.pretrained_embeds_path, 'rb') as fopen:
        for l in fopen:
            if idx < rand_num_line:
                idx += 1;
                continue;
            else:
                line_splits = l.decode().split();
                word = line_splits[0];
                expected_embeds = np.array(line_splits[1:]).astype(np.float32);
                print('word: ', word);
                #print('expected embeddings: ', expected_embeds);
                res_embeds = embeds.embeddings[embeds.word2idx[word]];
                #print('answer embeddings: ',  res_embeds);
                print('check embedings length: ', len(res_embeds) == len(expected_embeds));
                for i in range(len(res_embeds)):
                    print(math.isclose(expected_embeds[i], res_embeds[i], rel_tol = 0.00001), end = ' ');
                break;


if __name__ == '__main__':
    #test_assign_embeddings();
    #test_run_pipeline();
    test_embeddings();
    #run_pipeline_out();
    #write_glove_embeddings_bcolz();