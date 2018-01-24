from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import os;
from six.moves.urllib.request import urlopen;

import numpy as np;
import tensorflow as tf;

# Data sets
#IRIS_TRAINIG = '/Users/penpen926/workspace/MachineLearningPlayground/data/iris_training.csv';
IRIS_TRAINING ='c:/workspace/MachineLearningPlayground/data/iris_training.csv';
IRIS_TRAINING_URL= "http://download.tensorflow.org/data/iris_training.csv";

#IRIS_TEST = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris_test.csv";
IRIS_TEST = "c:/workspace/MachineLearningPlayground/data/iris_test.csv";
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv";

iris_model_dir = "c:/workspace/MachineLearningPlayground/model/iris_model"
def main():

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(\
        filename = IRIS_TRAINING,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TEST,
        target_dtype = np.int,
        features_dtype = np.float32
    )
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    classifier.train(input_fn = train_input_fn, steps = 2000);

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': np.array(test_set.data)},
        y = np.array(test_set.target),
        num_epochs = 1,
        shuffle= False
    )

    accuracy_score = classifier.evaluate(input_fn = test_input_fn)["accuracy"];
    print("Test accuracy:  ", accuracy_score);

    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x':new_samples},
        num_epochs = 1,
        shuffle = False
    )

    predictions = list(classifier.predict(input_fn = predict_input_fn));
    predicted_classes = [p['classes'] for p in predictions];
    print(predicted_classes);


main();