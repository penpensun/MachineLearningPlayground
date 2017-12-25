'''
Created on 21.12.2017

@author: GGTTF
'''

import os;
import pandas as pd;
import numpy as np;

# Test the functions of Dataframe. Learn Dataframe selecting, slicing and indexing
def testDataframe():

    # Generate dataframe
    dates = pd.date_range('1/1/2010',periods = 8);
    df = pd.DataFrame(np.random.randn(8,4), index = dates, columns=['A','B','C','D']);
    #print(df);
    panel  = pd.Panel({'one':df, 'two':df-df.mean()});
    #print(panel);
    
    
    #print('indexing');
    s = df[['A','B']];
    #print("s.A");
    #print(s.A)
    #print('df.B');
    #print(df.B);
    #print(s);
    #print('slicing');
    
    #print(df[1:4]);
    df1 = pd.DataFrame(np.random.randn(6,4), index = list(range(0,12,2)), columns = list(range(0,8,2)));
    #print(df1);
    #print(df1);
    #dfslice1 = df1.iloc[:2];
    #print(dfslice1);
    
    #dfslice2 = df1.iloc[1];
    #print(dfslice2);
    
    #dfsliceempty = df1.iloc[33:34];
    #print(dfsliceempty);
    
    #print('true false indexing');
    #seriestruefalse = pd.Series([True,True,False,False,True,False], index = list(range(0,12,2)) );
    #print(seriestruefalse);
    #dftruefalse = df1[seriestruefalse];
    #print(dftruefalse);
    #print(df1[lambda df: df>1]);
    
    #df2 = pd.DataFrame([[3,2,4],[3,3,2],[1,2,3]], index = list('abc'), columns = list('def'));
    #print(df2);
    
    #df2_selected = df2['e'].isin([2]);
    #print(df2_selected);
    
    
    df1 = pd.DataFrame(np.random.randn(6,4), index = list('abcdef'), columns = list('ABCD'));
    print('dataframe 1');
    print(df1);
    
    df_lambda1 = df1. loc[lambda df: df.A>0, :];
    print('lambda1');
    print(df_lambda1);
    
    df1_sample1 = df1.sample(3);
    print('sampling');
    print(df1_sample1);
    
    s1 = pd.Series([1,3,4,5,6,7,10], index = list('abcdefg'));
    print('Series s1:');
    print(s1);
    
    print('Enlargement by indexing');
    s1['i'] = 12;
    print(s1);
    
    print('quick accessing and setting');
    print('quick accessing:');
    print(s1.iat[2]);
    print(df1.at['b','B']);
    print(df1.iat[2,3]);
    
    
    print('Boolean indexing');
    print(df1[df1['A']>0]);
    
    
    
    
testDataframe();