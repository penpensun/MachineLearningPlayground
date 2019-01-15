import pandas as pd;
import matplotlib.pyplot as plt;

#input_exp1_pi2 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi2.xlsx";
input_exp1_pi2 = "/Users/penpen926/workspace/data/cnfuv/Data_Experiment_1/pi2.xlsx";

#input_exp1_pi3 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi3.xlsx";
input_exp1_pi3 = "/Users/penpen926/workspace/data/cnfuv/Data_Experiment_1/pi3.xlsx";

#input_exp1_pi4 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi4.xlsx";
input_exp1_pi4 = "/Users/penpen926/workspace/data/cnfuv/Data_Experiment_1/pi4.xlsx";
input_exp1_pi5 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi5.xlsx";

def view_dataset():
    df_exp1_pi2 = pd.read_excel(input_exp1_pi2);
    #print(df_exp1_pi2);
    #print(df_exp1_pi2.describe());
    fig = plt.figure();
    ax1 = fig.add_subplot(111);
    ax1.set_title("temperature vs. humidity");
    plt.xlabel("temperature");
    plt.ylabel("humidity");
    temperature_values = pd.to_numeric(df_exp1_pi2['Temperature'], errors='coarse');
    humidity_values = pd.to_numeric(df_exp1_pi2['Humidity'], errors='coarse');
    print('Temperature values:');
    print(temperature_values);
    print('Humidity:');
    print(humidity_values);
    plt.scatter(temperature_values, humidity_values);
    #plt.show();
    plt.savefig('tempature_vs_humidity.png');

def view_dataset2():
    df_exp1_pi2 = pd.read_excel(input_exp1_pi2);
    df_exp1_pi2['parsed_timepoints'] = parse_timepoint(df_exp1_pi2);
    temperature_values = pd.to_numeric(df_exp1_pi2['Temperature'], errors='coarse');
    humidity_values = pd.to_numeric(df_exp1_pi2['Humidity'], errors='coarse');
    timepoints = df_exp1_pi2['parsed_timepoints'].values;
    figure = plt.figure(num=1, figsize=(12,6));
    ax1 = figure.add_subplot(121);
    plt.ylabel('timepoints');
    plt.xlabel('temperature');
    ax1.scatter(timepoints, humidity_values);
    ax2 = figure.add_subplot(122);
    plt.ylabel('timepoints');
    plt.xlabel('temperature');
    ax2.scatter(timepoints, temperature_values);
    plt.savefig('timepoints_vs_temperature_humidity.png');

def read_exp1_pi2():
    df_exp1_pi2 = pd.read_excel(input_exp1_pi2);
    df_exp1_pi2['parsed_timepoints'] = parse_timepoint(df_exp1_pi2);
    df_exp1_pi2['temperature_values'] = pd.to_numeric(df_exp1_pi2['Temperature'], errors='coarse');
    df_exp1_pi2['humidity_values'] = pd.to_numeric(df_exp1_pi2['Humidity'], errors='coarse');
    return df_exp1_pi2;

def parse_timepoint(df):
    time_series = df['time'];
    parsed_time_series = time_series.map(compute_seconds);
    return parsed_time_series;


def compute_seconds(time_point_str):
    time_split = time_point_str.split(' ')[1].split(':');
    #print(time_split);
    return int(time_split[0])*3600 + int(time_split[1])*60 + float(time_split[2]);

if __name__== '__main__':
    #view_dataset();
    #view_dataset2();
    #parse_timepoint();
    #parse_timepoint();
    view_dataset2();
