import tensorflow as tf;
import pandas as pd;
import numpy as np;
from sklearn.model_selection import KFold;
from sklearn.preprocessing import StandardScaler;

FLAGS = tf.flags.FLAGS;

tf.flags.DEFINE_string('input_file_path', 'C:/workspace/programmes/data/PRSA_data_2010.1.1-2014.12.31.csv', '');
tf.flags.DEFINE_integer('num_layer', 3, '');
tf.flags.DEFINE_integer('num_hidden_units', 5, '');

def read_input ():
    input_data = pd.read_csv(FLAGS.input_file_path);
    return input_data;

def cv_split (input_data):
    kf_cv = KFold(n_splits = 5);
    print(input_data.columns.values)
    print('k folder cross validation');
    print(kf_cv)
    # define the x as input
    X = input_data[['No', 'year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']].values;
    y = input_data['pm2.5'].values
    test_arr = [True, True, False, True]
    test_input = input_data[:4];
    print(test_input[test_arr])

def filter_off_null (input_data):
    not_filter_off = []
    for index, row in input_data.isnull().iterrows():
        #print(row);
        #print(type(row));
        a = row.values;
        if any(row.values):
            not_filter_off.append(False); # If there exists true, indicating there is NaN in the row, then filter off
        else:
            not_filter_off.append(True); # If there is no true, indicating no NaN, then not filter off
    return input_data[not_filter_off];


def standardize (data):
    scaler = StandardScaler();
    scaler.fit(data)

if __name__ == '__main__':
    #cv_split(read_input())
    non_null_input_data = filter_off_null(read_input());
    print(non_null_input_data);


    #print(read_input());