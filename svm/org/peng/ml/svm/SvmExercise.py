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

os.chdir("/Users/penpen926/workspace/MachineLearningPlayground/data/")

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


def gaussianKernal(x1,x2,sigma):
    return np.exp(-( np.sum( (x1-x2)**2  ) /(2*(sigma**2)) ));


rawData = loadmat('./ex6data2.mat');
data  = pd.DataFrame(rawData['X'],columns=['X1',"X2"]);
data['y'] = rawData['y'];
positive = data[data['y'].isin([1])];
negative = data[data['y'].isin([0])];

fig,ax = plt.subplots(figsize=(12,8));
ax.scatter(positive['X1'],positive['X2'],s=30,marker='x',label='Positive');
ax.scatter(negative['X1'],negative['X2'],s=30,marker='o',label='Negative');
ax.legend();
plt.savefig("gaussian_kernel.png");
plt.close();


svc = svm.SVC(C=100, gamma=10, probability=True);
svc.fit(data[['X1','X2']],data['y']);
data['Probability'] = svc.predict_proba(data[['X1','X2']])[:,0];

fig,ax = plt.subplots(figsize=(12,8));
ax.scatter(data['X1'],data['X2'],s=30,c=data['Probability'],cmap="Reds");
plt.savefig("gaussian_kernel_2.png");
plt.close();


