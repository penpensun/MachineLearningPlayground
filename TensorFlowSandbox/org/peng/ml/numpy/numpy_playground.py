import numpy as np;

def test_ndarray():
    arr = np.random.randn(10);
    print(arr);
    print(type(arr));
    print("dim of the array.");
    print(arr.ndim);
    print("shape of the array.");
    print(arr.shape);
    nd_arr = np.ndarray(shape = (2,5), buffer = arr);
    print("ndarray");
    print(nd_arr);

    print("Zero ndarray.");
    zero_nd_arr = np.zeros(shape = (2,3,4));
    print(zero_nd_arr);

    print("one ndarray.");
    one_nd_arr = np.ones(shape = (2,3,4));
    print(one_nd_arr);

    print("empty ndarray.");
    empty_nd_arr = np.empty(shape = (2,3,4));
    print(empty_nd_arr);

    print("zero like array.");

    zero_like_ndarr = np.zeros_like(nd_arr);
    print(zero_like_ndarr);

test_ndarray();