import tensorflow as tf;
import numpy as np;
import pandas as pd;


iris_data = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris/iris.data";
model_dir = "/Users/penpen926/workspace/MachineLearningPlayground/model/"

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
    label = data.as_matrix(columns = ['class_num'])

    #create featured column
    feature_columns = [tf.contrib.layers.real_valued_column(column_name = "", dimension = 4)];
    model = tf.contrib.learn.DNNClassifier(hidden_units= [10,10,10],
                                       feature_columns = feature_columns,
                                       model_dir = model_dir,
                                       n_classes = 3); #Create the model
    train_the_model = model.fit(x = feature_matrix, y = label, max_steps = 10000);

    evalute_the_model = model.evaluate(x = feature_matrix, y = label);
    print(evalute_the_model);

dnn_classifier();