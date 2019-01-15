from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import f1_score;
from cnfuv_regress import read_exp1_pi2;

def gen_datasets(x,y):
    #Gerenate train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state=23);
    return (x_train, x_test, y_train,y_test);

def model1():
    dataset = read_exp1_pi2();
    # Use humidity and timepoints to predict temperature
    x = dataset[['humidity_values', 'parsed_timepoints']];
    y = dataset['temperature_values'];
    #print('x datset');
    #print(x);
    #print('y dataset');
    #print(y);
    print('split dataset:');
    x_train, x_test, y_train, y_test = gen_datasets(x, y);
    print('remove nan value');
    
    print('build up linear regression');
    reg = LinearRegression(normalize=True);
    reg.fit(x_train,y_train);
    print('the linear model:');
    print(reg);



if __name__ == '__main__':
    model1();


