import tensorflow as tf;
import numpy as np;
import pandas as pd;

iris_data = "/Users/penpen926/workspace/MachineLearningPlayground/data/iris/iris.data";

def learn_data_api():
    raw_data = np.matrix(data = [[1,2,3],[1,2,3],[1,2,3]], dtype = np.float32);
    data = tf.data.Dataset.from_tensor_slices(raw_data);
    iterator = data.make_one_shot_iterator();
    one_element = iterator.get_next();
    with tf.Session() as sess:
        print(sess.run(one_element));


def learn_iris_dataset():
    iris_df = pd.read_csv(iris_data, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','class']);
    print(iris_df);


#learn_data_api();
learn_iris_dataset();