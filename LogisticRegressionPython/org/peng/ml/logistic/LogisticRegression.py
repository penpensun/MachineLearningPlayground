'''
Created on 19.12.2017

@author: GGTTF
'''
# Implement the logistic regression
import numpy as np;
import pandas as pd;
import os;
import scipy.optimize as opt;
import matplotlib.pyplot as plt;

weights = [0.00,0.00,0.00];
alpha = 0.01;


def sigmoid(x):
    return 1/(1+np.exp(-x));

def sigmoid2(z):
    return 1/(1+np.exp(-z));


# X is a n*m matrix
def gradient1(weights, X,y):
    weights = np.matrix(weights);
    outputs = np.matrix(y);
    diff1 = X.T* outputs.T;
    diff1 = np.ravel(diff1);
    diff2 = np.zeros(X.shape[1]);
    for k in range(X.shape[1]):
        tempArr = np.zeros(X.shape[0]);
        for i in range(X.shape[0]):
            tempArr[i] = -sigmoid( weights*X[i,:].T )*X[i,k];
        diff2[k] = sum(tempArr);
    diff = diff1+diff2;
    return -diff/len(y);


def cost(weights,X,y):
    weights = np.matrix(weights);
    costArr = np.zeros(X.shape[0]);
    for i in range(len(costArr)):
        costArr[i] = -y[i]*np.log(sigmoid(weights*X[i,:].T))- \
            (1-y[i])*np.log(1- sigmoid(weights*X[i,:].T));
    return np.sum(costArr)/len(y);



def costReg(weights, X, y, learningRate):
    weights = np.matrix(weights);
    costArr = np.zeros(X.shape[0]);
    for i in range(len(costArr)):
        costArr[i] = -y[i]*np.log(sigmoid(weights*X[i,:].T))- \
            (1-y[i])*np.log(1- sigmoid(weights*X[i,:].T));
            
    penalty = np.sum(weights*weights.T)*learningRate/(len(y)*2);
    return np.sum(costArr)/len(y)+penalty;

def logistic_exercise():
    # change working folder
    #os.chdir("c:/workspace/MachineLearningPlayground/data/");
    os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/");
    datafile= "ex2data1.txt";
    # Read the data file
    df = pd.read_csv(datafile,sep=",",names=['exam1','exam2','pass']);
    
    #Draw the picture
    print(type(df['pass'].isin([1])));
    positives = df[df['pass'].isin([1])];
    negatives = df[df['pass'].isin([0])];
    
    fig,ax = plt.subplots(figsize=(12,8));
    ax.scatter(positives['exam1'],positives['exam2'],s = 50, c='b', marker = 'o', label= 'Passed');
    ax.scatter(negatives['exam1'],negatives['exam2'],s = 50, c = 'r', marker = 'x', label = "Failed");
    ax.legend();
    ax.set_xlabel('Exam 1 Score');
    ax.set_ylabel("Exam 2 Score");
    plt.savefig("logistic.png");
    #plt.show();
    
    
    #print(df);
    datamatrix = np.matrix(df);
    #print(datamatrix);
    #print(datamatrix.shape);
    
    intersectCol = np.ones((datamatrix.shape[0],1));
    inputmatrix = np.append(datamatrix[:, 0:2],intersectCol,axis=1 );
    outputs = np.ravel(datamatrix[:,2]);
    #Use opt fmin_tnc to optimize
    result = opt.fmin_tnc(func =cost , fprime = gradient1, x0 = weights, args=(inputmatrix,outputs));
    learningRate =1 ;
    #result = opt.fmin_tnc(func = costReg, fprime = gradient1, x0 = weights, args=(inputmatrix,outputs,learningRate));
    print('result');
    print(result[0]);
    print('cost after optimization:')
    print(cost(result[0],inputmatrix,outputs));
    testCostInput = inputmatrix;
    #print('testcostinput');
    #print(testCostInput);
    print('cost');
    print(cost(weights,testCostInput,outputs));
    
    
    #Prediction
    weights_opt = result[0];
    predictions = predict(weights_opt,inputmatrix);
    print('predictions');
    print(predictions);
    print(predictions == outputs);
    predict_result = (predictions == outputs)
    print('Accuracy');
    print(len(predict_result[predict_result == True])/ len(predict_result));



    
    
    #a = np.matrix([[1,2,5],[3,4,6]]);
#print(a[0,1]);
#print(a[:,1]);
#print(a.shape);
#print(a.shape[1]);

def predict(weights, X):
    prob = sigmoid(weights*X.T);
    return [1 if x>=0.5 else 0 for x in np.ravel(prob)];

logistic_exercise();

