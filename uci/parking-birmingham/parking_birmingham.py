import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy as sp;

data_file = '/Users/penpen926/workspace/data/uci-parking-birmingham/dataset.csv';

data = pd.read_csv(filepath_or_buffer = data_file);
print(data.columns.values);
def convert_time(x):
    str_time = x.split(' ')[1];
    return int(str_time.split(':')[0]) + float(str_time.split(':')[1])/60.0;

def visual_data(x):
    fig = plt.figure();
    ax1 = fig.add_subplot(111);
    ax1.set_title('Birmingham parking BHMBCCPST01');
    plt.xlabel('time');
    plt.ylabel('occupation');
    ax1.scatter(x['time'], x['Occupancy'])
    plt.show();


str_date = data['LastUpdated'].values[0];
#print(data['LastUpdated'].map(convert_time));

data['time'] = data['LastUpdated'].map(convert_time);
#print(data.where(data['time'] <10));
print(pd.unique(data['SystemCodeNumber']));
print(data[data['SystemCodeNumber'] == 'BHMBCCPST01']);
visual_data(data[data['SystemCodeNumber'] == 'BHMBCCPST01']);
