'''
Created on 19.12.2017

@author: GGTTF
'''
# Implement the logistic regression
import numpy as np;
import pandas as pd;
import os;
import scipy.optimize as opt;

weights = [0.001,0.001,0.001];
alpha = 0.01;

def sigmoid(weights, x):
    xFlattened = np.ravel(x);
    innerProd = np.dot(weights,xFlattened);
    #print("innerprod");
    #print(innerProd);
    #print("1+exp(innerprod)");
    #print(1+np.exp(-innerProd));
    #print(np.exp(-innerProd))
    return 1/(1+np.exp(-innerProd));

def sigmoid2(z):
    return 1/(1+np.exp(-z));


# X is a n*m matrix
def gradient1(weights, X,y):
    # Calculate the first part of difference
    # diff1 = sum( y_i *x_i.t * x_(ik))
    outputs = np.matrix(y);
    diff1 = X.T* outputs.T;
    diff1 = np.ravel(diff1);
    #print('diff1:');
    #print(diff1);
    # Calculate the second part of difference
    diff2 = np.zeros(X.shape[1]);
    for k in range(X.shape[1]):
        tempArr = np.zeros(X.shape[0]);
        for i in range(X.shape[0]):
            tempArr[i] = -sigmoid(weights,X[i,:])*X[i,k];
        
        diff2[k] = sum(tempArr);
    
    #print('diff2:');
    #print(diff2);
    
    diff = diff1+diff2;
    #print('grad1');
    #print(diff);
    
    return diff/len(y);


def gradient2(weights, X,y):
    weights = np.matrix(weights);
    
    parameters = int(weights.ravel().shape[1]);
    grad = np.zeros(parameters)

    error = sigmoid2(X * weights.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    #print('gradient2');
    #print(grad);
    return grad

def cost1(weights,X,y):
    costArr = np.zeros(X.shape[0]);
    for i in range(len(costArr)):
        costArr[i] = -y[i]*np.log(sigmoid(weights,X[i,:]))- \
            (1-y[i])*np.log(1- sigmoid(weights,X[i,:]));
    
    print('cost1 array:');
    print(costArr);
    print('cost1');
    print(np.sum(costArr)/len(y));
    return np.sum(costArr)/len(y);


def cost2(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid2(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid2(X * theta.T)))
    
    print('cost2 array:');
    print(first-second);
    
    print('cost2');
    print(np.sum(first - second)/(len(X)) );
    return np.sum(first - second) / (len(X));

def logistic_exercise():
    # change working folder
    os.chdir("c:/workspace/MachineLearningPlayground/data/");
    datafile= "ex2data1.txt";
    # Read the data file
    df = pd.read_csv(datafile,sep=",",names=['exam1','exam2','pass']);
    #print(df);
    datamatrix = np.matrix(df);
    #print(datamatrix);
    #print(datamatrix.shape);
    
    intersectCol = np.ones((datamatrix.shape[0],1));
    inputmatrix = np.append(datamatrix[:, 0:2],intersectCol,axis=1 );
    outputs = np.ravel(datamatrix[:,2]);
    #print('outputs:');
    #print(outputs);
    #print(inputmatrix);
    
    testInputArr = np.ravel(inputmatrix[1,:]);
    #print(testInputArr);
    #print(sigmoid(weights,testInputArr));
    
    #print("Gradient:");
    #print(gradient(weights,inputmatrix,outputs));
    

    #print("Cost:");
    #print(cost(weights,inputmatrix,outputs));
    
    
    #Use opt fmin_tnc to optimize
    
    #result = opt.fmin_tnc(func =cost1 , fprime = gradient1, x0 = weights, args=(inputmatrix,outputs));
    
    #print('After optimization.');
    #print(result[0]);
    
    #print("Cost after optimization:");
    #print(cost1(result[0],inputmatrix,outputs));
    
    testCostInput = inputmatrix[1:6,:];
    print('testcostinput');
    print(testCostInput);
    print('cost1');
    cost1(weights,testCostInput,outputs[1:6]);
    print('cost2');
    cost2(weights,testCostInput,outputs[1:6]);
    
#a = np.matrix([[1,2,5],[3,4,6]]);
#print(a[0,1]);
#print(a[:,1]);
#print(a.shape);
#print(a.shape[1]);

logistic_exercise();
