import tensorflow as tf;
import numpy as np;
def play_with_dataset():
    #按行切割
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]));
    dataset_random = tf.random_uniform([4,10]);
    print(dataset1.output_types);
    print(dataset1.output_shapes);

    with tf.Session() as sess:
        print(sess.run(dataset_random));


    dataset2 = tf.data.Dataset.from_tensor_slices(
        (tf.random_uniform([4]),
         tf.random_uniform([4,100], maxval = 100, dtype = tf.int32)));
    print(dataset2.output_types);
    print(dataset2.output_shapes);

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2));
    print(dataset3.output_types);
    print(dataset3.output_shapes);

    dataset4 = tf.data.Dataset.from_tensor_slices(
        {
            "a": tf.random_uniform([4]),
            "b": tf.random_uniform([4,100],maxval = 100, dtype = tf.int32)
        }
    )
    print(dataset4.output_types);
    print(dataset4.output_shapes);


def play_with_dataset2():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]));
    iterator = dataset.make_one_shot_iterator();
    one_element = iterator.get_next();
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element));
        except tf.errors.OutOfRangeError:
            print("end!");

def play_with_map():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5]));
    dataset = dataset.map(lambda x: x+1);

    iterator = dataset.make_one_shot_iterator();
    one_element = iterator.get_next();

    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element));
        except tf.errors.OutOfRangeError:
            print('end!');

    #print(dataset);

def play_with_batch():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]));
    dataset = dataset.batch(3);
    iterator = dataset.make_one_shot_iterator();
    one_element = iterator.get_next();

    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element));
        except tf.errors.OutOfRangeError:
            print("end!");

def play_with_shuffle():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6,7,8,9,10]));
    dataset = dataset.shuffle(buffer_size = 10000);

    iterator = dataset.make_one_shot_iterator();
    one_element = iterator.get_next();

    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element));
        except tf.errors.OutOfRangeError:
            print("end!");

def parse_function(filename, label):
    image_string = tf.read_file(filename);
    image_decoded = tf.image.decode_image(image_string);
    print('image decoded.');
    with tf.Session() as sess:
        print(sess.run(image_decoded));
    image_resized = tf.image.resize_images(image_decoded, [30,30]);
    return image_resized,label;

def read_in_figures():
    filenames = tf.constant([
        "c:/workspace/MachineLearningPlayground/data/poland_1.jpg",
        "c:/workspace/MachineLearningPlayground/data/poland_2.jpg"
    ]);

    labels = tf.constant(['poland1','poland2']);
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels));
    dataset = dataset.map(parse_function);

    iterator = dataset.make_one_shot_iterator();
    first_element = iterator.get_next();

    with tf.Session() as sess:
        print(sess.run(first_element));


def play_with_numpy_input_fn():
    age = np.arange(4)*10;
    print(age);
    height = np.arange(32,36);
    x = {'age': age, 'height': height};
    y = np.arange(-32,-28);

    with tf.Session() as sess:
        input_fn =  tf.estimator.inputs.numpy_input_fn(x, y,shuffle = False, batch_size = 2, num_epochs=1);

        features, targets = input_fn();
        print(sess.run(features));
        print(sess.run(targets));

#play_with_dataset();
#play_with_map();
#play_with_batch();
#play_with_shuffle()
#read_in_figures();
play_with_numpy_input_fn();