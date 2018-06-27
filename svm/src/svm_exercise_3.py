import pandas as pd;
import numpy as np;
import sklearn.svm as svm;
import scipy.io as scio;
import os;
import tensorflow as tf;
import matplotlib.pyplot as plt;

data_file = "data/ex6data2.mat";
os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/");

def compute_score_exercise():
    raw_data = scio.loadmat(data_file);
    #print(raw_data);
    #print(type(raw_data));
    X = raw_data['X'];
    y = raw_data['y'];

    #print(X);
    print(X.shape);
    print(y.shape);

    svc = svm.SVC(C = 20.0,kernel = 'rbf', gamma = 10);
    svc.fit(X,y);

    #Print the training accuracy

    score = svc.decision_function(X);
    label = svc.predict(X);
    print("The score:");
    #print(score);
    print(score.shape);
    print(score.ndim);

    print("The prediction:");
    #print(label);
    print(label.shape);
    print(label.ndim);

    score_label_mtx = np.matrix(np.zeros(shape = (863,2)));
    for i in range(len(score_label_mtx)):
        score_label_mtx[i,0] = score[i];
        score_label_mtx[i,1] = label[i];

    #print(score_label_mtx[240:300,:]);


    #Print out the accuracy
    print("The accuracy on training data:");
    print(svc.score(X,y));

    # The pred_correct array
    pred_correct = np.array(np.zeros(len(label)));
    for i in range(len(pred_correct)):
        pred_correct[i] = (label[i] == y[i]);

    #pred_correct = pred_correct.flatten();
    print("pred_correct.");
    print(pred_correct);

    print("lengths:");
    print(pred_correct.shape);
    print(X[:,0].shape);
    print(y.shape);
    print(label.shape);

    #Plot the scatter
    result_df = pd.DataFrame(data = {'X1': X[:,0],\
            'X2': X[:,1], 'predict_label': label, \
            'actual_label': y.flatten(),\
            'pred_correct': pred_correct});
    print("result_df");
    #print(result_df);


    #Plot the scatter graph

    correct_preds = result_df[result_df['pred_correct'].isin([1])];
    incorrect_preds = result_df[result_df['pred_correct'].isin([0])];
    correct_preds_positives = correct_preds[correct_preds['actual_label'].isin([1])];
    correct_preds_negatives = correct_preds[correct_preds['actual_label'].isin([0])];

    #print(incorrect_preds);
    #print(correct_preds_positives);
    #print(correct_preds_negatives);

    figure,ax = plt.subplots(figsize= (12,8));

    ax.scatter(correct_preds_positives['X1'], correct_preds_positives['X2'], s = 30, c = 'r');
    ax.scatter(correct_preds_negatives['X1'], correct_preds_negatives['X2'], s = 30, c = 'b');
    ax.scatter(incorrect_preds['X1'], incorrect_preds['X2'], s = 30, c = 'y');
    plt.savefig("svc_figure.png");

def compute_score_exercise_2():
    raw_data = scio.loadmat(data_file);
    print(type(raw_data));
    x = raw_data['X'];
    y = raw_data['y'];

    #Create the svc
    svc = svm.SVC(C = 20.0, gamma = 10.0);
    svc.fit(x,y);

    #Make predictions
    preds = svc.predict(x);

    #Compute the accruacy
    print("Accuracy on training data:");
    print(svc.score(x,y));

    #Plot the data
    label_compare = np.array(np.zeros(len(preds)));
    for i in range(len(label_compare)):
        if preds[i] == y[i]:
            label_compare[i] = 1.0;
        else:
            label_compare[i] = 0.0;

    result_df = pd.DataFrame(data = {'x1': x[:,0], 'x2': x[:,1],\
                                     'labels': y.flatten(), \
                                     'pred_labels': preds,\
                                     'is_correct': label_compare});

    correct_pred = result_df[result_df['is_correct'].isin([1.0])];
    incorrect_pred = result_df[result_df['is_correct'].isin([0.0])];

    correct_pred_positive = correct_pred[correct_pred['labels'].isin([1.0])];
    correct_pred_netagive = correct_pred[correct_pred['labels'].isin([0.0])];

    fig,ax = plt.subplots(figsize = (16,12));
    ax.scatter(correct_pred_positive['x1'], correct_pred_positive['x2'], s = 30, c = 'b');
    ax.scatter(correct_pred_netagive['x1'], correct_pred_netagive['x2'], s =30, c = 'r');
    ax.scatter(incorrect_pred['x1'], incorrect_pred['x2'], s = 30, c = 'y');

    plt.savefig("data/svc_plot.png");





def test_axis():
    const_matrix = tf.constant([[1.0,1.5],[1.0,1.5]]);
    const_mean_0 = tf.reduce_mean(const_matrix, axis=0);
    const_mean_1 = tf.reduce_mean(const_matrix, axis =1);

    with tf.Session() as sess:
        print("Matrix:");
        print(sess.run(const_matrix));
        print("axis = 0");
        print(sess.run(const_mean_0));
        print("axis = 1");
        print(sess.run(const_mean_1));


#test_axis();
compute_score_exercise_2();