import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import tensorflow as tf;

f = open("M:/Personal Data/projects/ai_materials/datasets/stock_datasets/dataset_1.csv");
#f = open("/home/ubuntu/workspace/dataset/stock_datasets/dataset_1.csv");


df = pd.read_csv(f, sep = ";");
data = np.array(df['highest price']);
data = data[::-1];

normalized_data = (data - np.mean(data))/np.std(data);
normalized_data = normalized_data[:, np.newaxis];

print(normalized_data);

