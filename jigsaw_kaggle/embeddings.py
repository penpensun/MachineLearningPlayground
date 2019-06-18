import bcolz;
import pickle;
import configuration as config;

if 'embeddings' not in globals():
    # read in embeddings
    embeddings = bcolz.open(config.bcolz_embedding_path, mode = 'r');
    #print('word2idx type: ', type(word2idx));

if 'word2idx' not in globals():
    # read in word2idx
    word2idx = pickle.load(open(config.word2idx_path, 'rb'));
