import collections;
import numpy as np;
import tensorflow as tf;

poetry_file = "c:/workspace/data/poetry.txt";
# Poetry collections
poetrys  = [];


def read_poetrys():
    global peotry_file;
    global poetrys;
    with open(poetry_file, "r", encoding = 'utf-8', ) as f:
        for line in f:
            try:
                title,content = line.strip().split(":");
                content = content.replace(' ',"");
                if '_' in content or '（' in content or '(' in content \
                        or "《" in content or '[' in content:
                    continue;
                if len(content) < 5 or len(content) >79:
                    continue;
                content = '['+content+']';
                poetrys.append(content);
            except Exception as e:
                pass


def preprocessing():
    #sort the poetries with number of characters
    global poetrys;
    poetrys = sorted(poetrys, key = lambda line: len(line));
    print ("Total number of poetries: %d",len(poetrys));

    #compute the occurrence of each character
    all_words = [];
    for poetry in poetrys:
        all_words += [word for word in poetry];
    counter = collections.Counter(all_words);
    count_pairs = sorted(counter.items(), key = lambda x: -x[1]);
    words, _ = zip (*count_pairs);
    # use the k most frequent words
    words = words[:len(words)]+(' ',);
    # map each word to an id
    word_num_map = dict(zip(words, range(len(words))));
    # Convert the word into one-hot vector
    to_num = lambda word: word_num_map.get(word, len(words));
    poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys];

    return poetrys_vector, word_num_map;

def gen_batch():
    global batch_size;
    global poetrys_vector;
    global word_num_map;

    n_chunk = len(poetrys_vector);
    x_batches =[];
    y_batches =[];

    for i in range (n_chunk):
        start_index = i*batch_size;
        end_index = start_index+batch_size;

        batches = poetrys_vector[start_index:end_index];
        length = max(map(len,batches));
        xdata = np.full((batch_size, length), word_num_map[' '], np.int32);
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row];

