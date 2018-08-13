import tensorflow as tf;
num_classes = 10;
embedding_size = 3;

def play_embedding():
    with tf.variable_scope('embedding'):
        inputs = [[1,4,6,7,9,0,2],\
            [2,3,5,8,0,3,4]];
        inputs_one_hot = tf.one_hot(inputs, depth=10);
        embedding = tf.get_variable('embedding', [num_classes, embedding_size]);

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        inputs_one_hot_eval = inputs_one_hot.eval();
        embedding_eval = embedding.eval();
        embedding_vec = tf.nn.embedding_lookup(params=embedding, ids=inputs);
        embedding_vec_eval = embedding_vec.eval();

        # print("Embedding:");
        #
        #
        # print(embedding_eval);
        #
        #
        # print("One hot eval:");
        # print(inputs_one_hot_eval);
        #
        # print("Look up in embedding matrix.");
        #
        # print(embedding_vec.eval());
        #
        # print("Result comparison:");
        # print("inputs[0]:", inputs[0]);
        # print("embedding vec:");
        # print(embedding_eval[inputs[0]]);
        # print("looked up embedding vec:");
        # print(embedding_vec_eval[0]);
        #
        # print("Result comparison:");
        # print("inputs[0]:", inputs[2]);
        # print("embedding vec:");
        # print(embedding_eval[inputs[2]]);
        # print("looked up embedding vec:");
        # print(embedding_vec_eval[2]);

        print("The shape of embedded vector.");
        print(embedding_vec_eval.shape);

        print("one_hot vector: ");
        print(inputs_one_hot_eval);
        print("one hot vector shape:");
        print(inputs_one_hot_eval.shape);

play_embedding();

