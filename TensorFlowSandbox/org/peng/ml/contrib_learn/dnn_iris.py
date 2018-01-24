from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import os;
import urllib;

import numpy as np;
import tensorflow as tf;

# Data sets
#IRIS_TRAINIG = '/Users/penpen926/workspace/MachineLearningPlayground/data/iris_training.csv';
IRIS_TRAINING ='c:/workspace/MachineLearningPlayground/data/iris_training.csv';
IRIS_TRAINING_URL= "http://download.tensorflow.org/data/iris_training.csv";

#IRIS_TEST = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris_test.csv";
IRIS_TEST = "c:/workspace/MachineLearningPlayground/data/iris_test.csv";
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv";


def main():

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(\
        filename = IRIS_TRAINIG,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TEST,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)];
    print("Training set");
    print(training_set);
    print(type(training_set));




main();