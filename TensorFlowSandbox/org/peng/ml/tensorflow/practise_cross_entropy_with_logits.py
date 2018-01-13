import tensorflow as tf;
import numpy as np;

def  practise_cross_entropy_with_logits():
    y_ = np.matrix(data =[[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0]]);
    y_ = y_.T;
    print("y matrix.");
    print(y_);
    output_arr = np.matrix(data = [gen_nrand(3),gen_nrand(3),gen_nrand(3),\
                                 gen_nrand(3),gen_nrand(3),gen_nrand(3)]);
    output_arr = output_arr.T;
    print("output array:");
    print(output_arr);



def gen_nrand(length):
    return np.random.randn(length);

practise_cross_entropy_with_logits();