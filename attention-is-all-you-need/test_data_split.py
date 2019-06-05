import pandas as pd;
from sklearn import datasets, linear_model;
from sklearn.model_selection import train_test_split;
import matplotlib.pyplot as plt;

data_set = datasets.load_diabetes();
#print(data_set);
#print(data_set['data'].shape);
#print(data_set['target'].shape);
columns = "age sex bmi map tc ldl hdl tch ltg glu".split() # Declare the columns names
#print(columns);
df = pd.DataFrame(data_set.data, columns = columns);
#print(df);
y = data_set.target;

# create training and testing vars
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.2);
#print(x_train.shape);
#print(y_train.shape);

# fit a linear model
lm = linear_model.LinearRegression();
lm_model = lm.fit(x_train, y_train);
predictions = lm_model.predict(x_test);
print(predictions);
print(y_test);

plt.scatter(y_test, predictions);
plt.xlabel('true values');
plt.ylabel('predictions');
plt.show()