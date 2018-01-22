import tensorflow as tf;
import numpy as np;

def  practise_cross_entropy_with_logits(y_, output_arr):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = output_arr);

    with tf.Session() as sess:
        print("softmax cross entropy.");
        return sess.run(softmax_cross_entropy);

def compute_softmax(array):
    if (np.ndim(array) == 1):
        return np.exp(array)/np.sum(np.exp(array));
    else:
        ans = np.ones(shape = array.shape);
        for i in range(len(ans)):
            ans[i,:] = np.exp(array[i,:])/np.sum(np.exp(array[i,:]))
        return ans;

def compute_softmax_cross_entropy(label, output):
    soft_max_out = compute_softmax(output);
    log_output= np.log(soft_max_out);
    print("output:");
    print(soft_max_out);
    print("log output:");
    print(log_output);
    res = np.array(np.zeros(len(label)));
    for i in range(len(label)):
        label_vec = label[i,:];
        log_output_vec = log_output[i,:];
        label_vec = np.squeeze(np.array(label_vec), axis = 0);


        res[i] = np.sum(-np.dot(label_vec, log_output_vec));
    return np.sum(res);



#This method compares the results of softmax_cross_entropy_with_logits and the
#self-computed softmax cross entropy with logits
def compare_softmax_cross_entropy():
    output_arr = np.matrix(data = [gen_nrand(3),gen_nrand(3),gen_nrand(3),\
                                 gen_nrand(3),gen_nrand(3),gen_nrand(3)]);

    y_ = np.matrix(data=[[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]);

    function_res = practise_cross_entropy_with_logits(y_, output_arr);
    self_computed_res = compute_softmax_cross_entropy(y_, output_arr);

    print("function results:");
    print(np.sum(function_res));
    print("self-computed results:");
    print(self_computed_res);


def compare_softmax():
    arr = np.matrix(data = [gen_nrand(3),gen_nrand(3),gen_nrand(3),\
                                 gen_nrand(3),gen_nrand(3),gen_nrand(3)]);
    with tf.Session() as sess:
        function_res = sess.run(tf.nn.softmax(logits = arr));

    self_computed_res = compute_softmax(arr);

    print("function results: ");
    print(function_res);
    print("self computed results: ");
    print(self_computed_res);


def gen_nrand(length):
    return np.random.randn(length);


def matrix_flatten():
    mtx = np.matrix(np.ones(shape = (2,2)));
    mtx_flatten = mtx.flatten();
    arr  = np.squeeze(np.array(mtx_flatten),axis  = 0);
    print(mtx);
    print(arr);
    print(type(arr));
    print(arr.shape);

    print(np.dot(arr, arr));

#practise_cross_entropy_with_logits();
compare_softmax_cross_entropy();
#compare_softmax();

#matrix_flatten();