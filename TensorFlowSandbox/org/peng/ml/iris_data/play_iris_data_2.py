import tensorflow as tf;
import pandas as pd
import numpy as np;

#iris_data = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris/iris.data";
iris_data = "c:/workspace/MachineLearningPlayground/data/iris/iris.data";
#model_dir = "/Users/penpen926/workspace/MachienLearningPlayground/data/model/";
model_dir = "c:/workspace/MachineLearningPlayground/data/model";

def dnn_classifier():
    raw_df = pd.read_csv(iris_data, names = ['sepal_length', 'sepal_width', 'pedal_length', 'pedal_width', 'class_str']);
    raw_df['class_num'] = np.zeros(shape = [len(raw_df)], dtype = np.int);

    # reset the index
    raw_df['index'] = range(len(raw_df));
    raw_df = raw_df.set_index(keys = 'index');
    for i in range(len(raw_df)):
        if raw_df.loc[i,'class_str'] == 'Iris-setosa':
            raw_df.loc[i,'class_num'] = 0;
        if raw_df.loc[i,'class_str'] == 'Iris-versicolour':
            raw_df.loc[i,'class_num'] = 1;
        if raw_df.loc[i,'class_str'] == 'Iris-virginica':
            raw_df.loc[i,'class_num'] = 2;

    #print(raw_df);
    feature_cols = [tf.contrib.layers.real_valued_column(column_name="", dimension = 4)]; # Generate feature_columns
    dnn_model = tf.contrib.learn.DNNClassifier(hidden_units = [10,20,5],
                                               feature_columns = feature_cols,
                                               model_dir = model_dir,
                                               n_classes = 3);
    data_features = np.matrix(data = raw_df.loc[:,['sepal_length', 'sepal_width', 'pedal_length','pedal_width']], dtype = np.float);
    data_labels = np.array(object = raw_df['class_num'], dtype = np.int);
    print(data_features);
    print(data_labels);
    dnn_model.fit(x = data_features, y = data_labels, steps = 10000)  # fit the model
    eval_res = dnn_model.evaluate(x = data_features, y=data_labels);
    print("evaluationg results:")
    print(eval_res);

    pred_classes = dnn_model.predict_classes(x = data_features);
    print("predict classes:");
    print(list(pred_classes));

    pred_proba = dnn_model.predict_proba(x = data_features);
    print("predict proba.");
    print(list(pred_proba));


dnn_classifier();