import tensorflow as tf;
import pandas as pd
import numpy as np;

iris_data = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris/iris.data";
model_dir = "/Users/penpen926/workspace/MachienLearningPlayground/data/model/"

def dnn_classifier():
    raw_df = pd.read_csv(iris_data, names = ['sepal_length', 'sepal_width', 'petal_length', 'pedal_wdith', 'class_str']);
    raw_df['class_num'] = np.zeros(shape = [len(raw_df)], dtype = np.int);

    for i in range(len(raw_df)):
