import tushare as ts;
import pandas as pd;

df = ts.get_hist_data('601398');
df.to_csv("/Users/penpen926/workspace/data/stock/601398.csv")
