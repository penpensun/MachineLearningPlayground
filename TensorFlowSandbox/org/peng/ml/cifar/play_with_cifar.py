import numpy as np;
import pickle;
import matplotlib.pyplot as pyplt;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.preprocessing import LabelBinarizer;
from sklearn.model_selection import train_test_split;
import tensorflow as tf;

def load_cifar10_batch(cifar10_dataset_folder, batch_id):

    with open (cifar10_dataset_folder+"/data_batch_"+str(batch_id), mode="rb") as file:
        batch = pickle.load(file, encoding = 'latin1');

    features = batch['data'].reshape( ( len(batch['data']),3,32,32)).transpose(0,2,3,1);
    labels = batch['labels']
    return features, labels;

def get_train_dataset():
    cifar10_path = "c:/workspace/data/cifar-10-batches-py/";
    x_train,y_train = load_cifar10_batch(cifar10_path,1);
    #load other dataset and concatenate to the first
    for i in range(2,6):
        features, labels = load_cifar10_batch(cifar10_path,i);
        x_train,y_train = np.concatenate([x_train, features]), np.concatenate([y_train, labels]);
    return x_train, y_train;

def get_test_dataset():
    cifar10_path = "c:/workspace/data/cifar-10-batches-py/";
    with open(cifar10_path+"test_batch", 'rb') as file:
        batch = pickle.load(file, encoding='latin1');
        x_test = batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1);
        y_test = batch['labels'];
        return x_test,y_test;


def show_pics():
    x_train, y_train = get_train_dataset();
    fig, axes = pyplt.subplots(nrows=3, ncols = 20, sharex=True, sharey=True, figsize= (80,12));
    imgs = x_train[:60];
    for image, row in zip([imgs[:20],imgs[20:40], imgs[40:60]], axes):
        for img, ax in zip (image, row):
            ax.imshow(img);
            ax.get_xaxis().set_visible(False);
            ax.get_yaxis().set_visible(False);

    fig.tight_layout(pad=0.1);
    pyplt.show();

def knn_classifier():
    x_train, y_train = get_train_dataset();
    x_test,y_test = get_test_dataset();

    x_train_rows = x_train.reshape(x_train.shape[0], 32*32*3);
    x_test_rows = x_test.reshape(x_test.shape[0], 32*32*3);
    minmax = MinMaxScaler();
    print("start transforming.")
    x_train_rows =  minmax.fit_transform(x_train_rows);
    x_test_rows = minmax.fit_transform(x_test_rows);


    k = [1,3,5];
    for i in k:
        model = KNeighborsClassifier(n_neighbors= i, algorithm='ball_tree', n_jobs = 6);
        model.fit(x_train_rows, y_train);
        print("Start predicting");
        preds = model.predict(x_test_rows);
        print("k = %s, Accuracy = %f"%(i, np.mean(y_test == preds)));



def gen_weights(weight_shape):
    weights = tf.truncated_normal(shape = weight_shape, stddev = 0.1 )
    return weights;

def cnn_classifier(n_class):
    x_train,y_train = get_train_dataset();
    x_test,y_test = get_test_dataset();
    minmax = MinMaxScaler();
    x_train_rows = x_train.reshape(x_train.shape[0],32*32*3);
    x_test_rows = x_test.reshape(x_test.shape[0], 32*32*3);

    #Nomralization
    minmax = minmax.fit(x_train); #because self is returned.
    x_train = minmax.transform(x_train_rows);
    x_test = minmax.transform(x_test_rows);
    #Reshape
    x_train = x_train.reshape(x_train[0],32,32,3);
    x_test = x_test.reshape(x_test[0],32,32,3);

    n_class = 10;
    lb = LabelBinarizer().fit(np.array(range(n_class)));
    x_test = lb.transform(x_test);
    y_test = lb.transform(y_test);

    train_ratio = 0.8;
    x_train,x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                     train_size = train_ratio,
                                                     random_state= 123);
    #Create convolutional network
    img_shape = x_train.shape;
    keep_prob = 0.6;
    epochs = 20;
    batch_size = 128;

    inputs = tf.placeholder(tf.float32, [None,32,32,3],name = 'inputs');
    targets = tf.placeholder(tf.float32, [None, n_class], name = 'targets');

    #First layer
    conv1_filter_shape = [4,4,3,32];
    conv1 = tf.nn.conv2d(input = inputs,
                         filter = gen_weights(conv1_filter_shape),
                         strids = [1,1,1,1],
                         padding = "SAME");
    conv1 = tf.nn.relu(features = conv1);
    #First max pooling layer
    max_pool1 = tf.nn.max_pool(value = conv1,
                               ksize = [1,2,2,1],
                               strides = [1,1,1,1],
                               padding = "SAME");
    #Second layter
    conv2_filter_shape = [4,4,32,64];
    conv2 = tf.nn.conv2d(input = inputs,
                         filter = gen_weights(conv2_filter_shape),
                         strieds = [1,1,1,1],
                         padding = "SAME");
    conv2 = tf.nn.relu(features = conv2);
    #Second max pooling layer, output shape [batch, 8,8,64]
    max_pool2 = tf.nn.max_pool(value = conv2,
                               ksize = [1,2,2,1],
                               strides = [1,1,1,1],
                               padding = "SAME");
    #First full connection layer
    fc1_weights_shape = [8*8*64,1024];
    fc1_weights = gen_weights(fc1_weights_shape);
    max_pool2 = tf.reshape(tensor = max_pool2, shape = [-1, 8*64*64]);
    fc1 = tf.matmul(a = max_pool2 , b = fc1_weights);

    #dropout layer
    dropout = tf.nn.dropout(x = fc1, keep_prob = keep_prob);

    #Second full connection layer
    fc2_weights_shape = [1024, n_class];
    fc2_weights = gen_weights(fc2_weights_shape);
    fc2 = tf.matmul(a = dropout, b = fc2_weights);

    #Get the output
    


def test():
    n_class = 10;
    a  = np.array(range(n_class));
    print(type(a));
    print(a);
#show_pics();
#knn_classifier();
test();


