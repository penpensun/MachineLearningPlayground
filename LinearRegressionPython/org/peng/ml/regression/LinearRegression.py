'''
Created on 09.11.2017

@author: Peng Sun
'''

import os;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

def computeCost(X,y,theta):
    inner = np.power((X*theta.T)-y,2);
    return np.sum(inner)/(2*len(X));


os.chdir("c:/workspace/MachineLearningPlayground/data/");
path = os.getcwd()+"/regression_data1.txt";

data = pd.read_csv(path, header= None, names= ['Population', 'Profit']);
data.head();
print(data.head());
print(data.describe());
#print(data.plot(kind='scatter', x='Population', y='Profit',figsize=(12,8)));
data.plot(kind='scatter', x='Population', y='Profit',figsize=(12,8));
#plt.show();

data.insert(0,"Ones",1);
cols = data.shape[1];

X = data.iloc[:, 0:cols-1];
Y = data.iloc[:, cols-1:cols];

X = np.matrix(X.values);
Y = np.matrix(Y.values);
theta = np.matrix(np.array([0,0]));

#print(computeCost(X,Y,theta));

print(theta.shape);

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape));
    parameters = int(theta.ravel().shap[1]);
    cost = np.zeros(iters);
    
    
    



