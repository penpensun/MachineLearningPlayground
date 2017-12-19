'''
Created on 2017年11月25日

@author: penpen926
'''

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sb;
from scipy.io import loadmat;
from sklearn import svm;
import pylab;
import os;

#os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/")
os.chdir("c:/workspace/MachineLearningPlayground/data/");


def svmEx6Data1():
    rawData = loadmat("ex6data1.mat");
    data = pd.DataFrame(rawData['X'], columns = ('x1','x2'));
    # Append the observations
    data['y'] = rawData['y'];
    
    # Init the positive and negative data
    positives = data[data['y'].isin([1])];
    negatives = data[data['y'].isin([0])];
    
    #Init the subplots
    fig,ax = plt.subplots(figsize=(12,8));
    
    # draw the positives
    ax.scatter(positives['x1'],positives['x2'],s=50, marker='o',label='Positives');
    ax.scatter(negatives['x1'],negatives['x2'],s=50, marker='x',label='Negatives');
    
    fig.savefig("linear_svc_pos_neg.png");
    
    svc = svm.LinearSVC(C=1000, loss='hinge',max_iter=1000);
    
    svc.fit(data[['x1','x2']],data['y']);
    
    data['SVM 1 Confidence'] = svc.decision_function(data[['x1','x2']]);
    print("SVM 1 Confidence");
    print(data['SVM 1 Confidence']);
    
    fig,ax = plt.subplots(figsize=(12,8));
    ax.scatter(data['x1'],data['x2'],s=50,c=data['SVM 1 Confidence'],cmap='seismic');
    ax.set_title("SVM Decision Confidence");
    plt.savefig('svm_confidence_linear_svc.png')
    
    
    svcScore = svc.score(data[['x1','x2']],data['y']);
    print(svcScore);


def svmEx6data2():
    #Read the raw data from mat file
    rawData = loadmat("ex6data2.mat");
    data = pd.DataFrame(rawData['X'], columns=['X1','X2']);
    print(type(rawData['X']));
    # Append on the result column
    data['Obs'] = rawData['y'];
    # Get the positives and negatives
    positives = data[data['Obs'].isin([1])];
    negatives = data[data['Obs'].isin([0])];
    
    #Visualize the data
    fig,ax = plt.subplots(figsize=(12,8));
    ax.scatter(positives['X1'],positives['X2'],s=50,marker="o");
    ax.scatter(negatives['X1'],negatives['X2'],s =50, marker = "x");
    
    plt.savefig("svm_pos_neg2.png")
    
    
    # set up an svm model using gaussian kernel
    svmClassifier = svm.SVC(C =100.0, gamma='auto',kernel='rbf', max_iter=1000,probability=True);
    svmClassifier.fit(data[['X1','X2']],data['Obs']);
    #data['Probability'] 
    x= svmClassifier.predict_proba(data[['X1','X2']]);
    data['Probability'] = x[:,0];
    fig,ax = plt.subplots(figsize=(12,8));
    ax.scatter(data['X1'],data['X2'],s=30, c=data['Probability'],cmap="Reds");
    plt.savefig("svm_rbf_class.png");
    #print(x);
    #print(type(x));


def svmEx6dat3():
    rawData= loadmat("ex6data3.mat");
    x = rawData['X'];
    xValidation = rawData['Xval'];
    y = rawData['y'].flatten();
    yValidation = rawData['yval'];
    
    cValues = [0.01, 0.03, 0.1, 0.3, 1,3,10,30,100];
    gammaValues = [0.01,0.03,0.1,0.3,1,3,10,30,100];
    
    bestScore = 0;
    bestParams = {'C':None, 'gamma':None};
    
    for c in cValues:
        for gamma in gammaValues:
            svc = svm.SVC(C=c, gamma = gamma);
            svc.fit(x,y);
            score = svc.score(xValidation,yValidation);
            if score > bestScore:
                bestScore = score;
                bestParams['C'] = c;
                bestParams['gamma'] = gamma;
    return bestScore,bestParams;
    
    
def gaussianKernel(X,Y,sigma):
    return np.exp(-(sum((X-Y)**2) / (sigma*sigma) ) );

bestScore,bestParams = svmEx6dat3();
print(bestScore);
print(bestParams);
#svmEx6data2();
    
