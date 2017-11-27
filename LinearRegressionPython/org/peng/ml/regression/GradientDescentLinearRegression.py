'''
Created on 09.11.2017

@author: Peng Sun
'''

import os;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import pylab;
import time;

def computeCost(X,y,theta):
    inner = np.power((X*theta.T)-y,2);
    return np.sum(inner)/(2*len(X));

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape));
    parameters = int(theta.ravel().shape[1]);
    cost = np.zeros(iters);
    
    for i in range(iters):
        error = (X*theta.T)-y;
        #if i ==1:
            #print("gradient descent 1.");
            #print(error[0:10,0]);
        for j in range(parameters):
            term = np.multiply(error,X[:,j]);
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term));
            if i==4:
                print("gradient descent 1.");
                print((alpha/len(X))*np.sum(term));
        
        theta= temp;
        if i==4:
            print("gradient descent 1, theta:");
            print(theta);
        cost[i] = computeCost(X,y,theta);
    
    return theta,cost;


def gradientDescent2(X,y,theta,alpha,iters):
    cost = np.zeros(iters);
    for i in range(iters):
        error = (X*theta.T) -y;

        theta = theta.T - alpha*(X.T * error)/len(X);
        theta = theta.T;
        #if i==4:
            #print("gradient descent 2.");
            #print(alpha*(X.T * error)/len(X));
            #print("gradient descent 2, theta: ");
            #print(theta);
        
        cost[i] = computeCost(X,y,theta);
    return theta, cost;
    

#os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/");
os.chdir("C:/workspace/MachineLearningPlayground/data/");
path = os.getcwd()+"/regression_data1.txt";

data = pd.read_csv(path, header= None, names= ['Population', 'Profit']);
data.head();
#print(data.head());
#print(data.describe());
#print(data.plot(kind='scatter', x='Population', y='Profit',figsize=(12,8)));
data.plot(kind='scatter', x='Population', y='Profit',figsize=(12,8));
plt.show();
pylab.show();


data.insert(0,"Ones",1);
cols = data.shape[1];

X = data.iloc[:, 0:cols-1];
Y = data.iloc[:, cols-1:cols];

X = np.matrix(X.values);
print("X");
print(X);
Y = np.matrix(Y.values);
print("Y");
print(Y);
print(np.shape(Y));


theta = np.matrix(np.array([0,0]));

#print(computeCost(X,Y,theta));
alpha = 0.01;
iters = 1000;
#g,cost = gradientDescent(X,Y,theta,alpha,iters);
#print("gradient descent 1, parameters:")
#print(g);
#print(g);
#print(cost);
#tempVector = np.matrix(np.array([1,1]));
#tempVector = tempVector.T;
#print(tempVector);
#print(type(tempVector));
#tempMultiVector = tempVector*2;
#print(tempMultiVector);

beforeRun = time.time();
for i in range(100):
    g,cost =gradientDescent2(X, Y, theta, alpha, iters);
print("gradient descent 2, parameters: ");
print(g);
afterRun = time.time();
print("Duration: ");
print(afterRun -beforeRun);






