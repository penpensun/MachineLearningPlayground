import tensorflow as tf;
import numpy as np;
import pandas as pd;


iris_data = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris/iris.data";

def dnn_classifier():
    data = pd.read_csv(iris_data, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','class_str']);
    data['class_num'] = np.zeros(shape = [len(data), ], dtype = np.int);
    #print(data);

    for i in range(len(data)):
        if data.iloc[i,:].loc['class_str'] == 'Iris-setosa':
            data.iloc[i,:] = data.iloc[i,:].set_value(label = 'class_num', value = 0) ;
        elif data.iloc[i,:].loc['class_str'] == 'Iris-versicolour':
            data.iloc[i,:]= data.iloc[i,:].set_value(label = 'class_num', value = 1);
        elif data.iloc[i,:].loc['class_str'] == 'Iris-virginica':
            data.iloc[i,:] = data.iloc[i,:].set_value(label = 'class_num', value =2);

    #Convert into np matrix
    feature_matrix = data.as_matrix(columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']);
    label = 


dnn_classifier();