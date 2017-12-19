'''
Created on 28.11.2017

@author: GGTTF
'''
import os;
from sklearn import svm;
import numpy as np;
import pandas as pd;
import scipy.io as scpio;
import matplotlib.pyplot as plt;

os.chdir("c:/workspace/MachineLearningPlayground/data/");
#Read the data
rawData = scpio.loadmat("ex6data2.mat");

#Convert the data into pandas.DataFrame
data = pd.DataFrame(rawData['X'], columns=['X1','X2']);

data['y'] = rawData['y'];

#Draw the positives and the negatives
positives = data[data['y'].isin([1])];
negatives = data[data['y'].isin([0])];


fig,ax = plt.subplots(figsize=(12,8));

ax.scatter(positives['X1'],positives['X2'], marker="o",s = 50);
ax.scatter(negatives['X1'],negatives['X2'], marker='x',s= 50);

#plt.show();

# Start building up the svm classifier using kernel of rbf

rbfSvc =svm.SVC(C=100, kernel='rbf', gamma=100,max_iter=1000);
rbfSvc.fit(data[['X1','X2']],data['y']);
data['Confidence'] =  rbfSvc.decision_function(data[['X1','X2']]);
print(data['Confidence']);
rbfSvcScore = rbfSvc.score(data[['X1','X2']],data['y']);
print(rbfSvcScore);









