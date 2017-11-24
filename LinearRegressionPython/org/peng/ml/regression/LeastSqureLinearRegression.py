'''
Created on 23.11.2017

@author: GGTTF
'''

import numpy as np;
import os;
import pandas as pd;


os.chdir("C:/workspace/MachineLearningPlayground/data/");
path = os.getcwd()+"/regression_data1.txt";
data = pd.read_csv(path,header= None, names= ['Population', 'Profit']);
data.insert(0,"Intersect",1);
numOfCols = data.shape[1];
predictorMatrix = data.iloc[:,0:numOfCols-1];
observedMatrix = data.iloc[:, numOfCols-1: numOfCols];

print("Predictor Matrix: ");
print(predictorMatrix);

print("Observed Matrix: ");
print(observedMatrix);



