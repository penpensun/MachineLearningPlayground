import pandas as pd;
import matplotlib.pyplot as plt;

input_exp1_pi2 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi2.xlsx";
input_exp1_pi3 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi3.xlsx";
input_exp1_pi4 = "C:\\workspace\\programmes\\data\\CNFUV_datasets\\Data_Experiment_1\\pi4.xlsx";
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
    temperature_values = pd.to_numeric(df_exp1_pi2['Temperature'], error='coarse');
    humidity_values = pd.to_numeric(df)
    plt.scatter(df_exp1_pi2['Temperature'].values, df_exp1_pi2['Humidity'].values);
    plt.show();


if __name__== '__main__':
    view_dataset();