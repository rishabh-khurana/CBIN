import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import json


def get_data(file):
    usecols=['age','sex','chest_pain','blood_pressure','serum_cholesterol','blood_sugar','electrocardiographic_result','max_heart_rate','exercise_induced','old_peak','slope','major_vessels','thal','target']
    df = pd.read_csv(file,sep=',',header=None ,names=usecols)
    
    df.replace('-', np.nan)
    return df

def scattered_json():
    df = get_data("processed.cleveland.data")
    scattered_json = [] 
    for col in df:
        if col != 'age' and col != 'sex':
            record = {}

            title = {}
            title['text'] = 'Age (x-axis) and ' + col.replace("_"," ").title()+' (y-axis)'
            record['title'] = title

    
            xAxis = {}
            
            xtitle = {}
            xtitle['text'] = 'Age'
            xAxis['title'] = xtitle

            record['xAxis'] = xAxis


            yAxis = {}
            
            ytitle = {}
            ytitle['text'] = 'Age (x-axis) and ' + col.replace("_"," ").title()+' (y-axis)'
            yAxis['title'] = ytitle

            record['yAxis'] = yAxis

            
            
            series = []

            male_series = {}
            male_series['name']= 'Male'
            male_series['colour']='rgba(119, 152, 191, .5)'            
            male_series['data'] = df.loc[df['sex'] == 1,['age',col]].values.tolist()
            series.append(male_series)

            female_series = {}
            female_series['name']= 'Female'
            female_series['colour']='rgba(223, 83, 83, .5)'
            female_series['data'] = df.loc[df['sex'] == 0,['age',col]].values.tolist()
            series.append(female_series)

            record['series'] = series

            #print(record)
            json_record = json.dumps(record)

            scattered_json.append(json_record)
            

            '''
            plt.scatter(x=male['age'],y=male[col],c='Blue', label='Male')
            plt.scatter(x=female['age'],y=female[col],c='Pink', label='Female')
            #plt.title('Age (x-axis) and ' + col.replace("_"," ").title()+' (y-axis)', fontsize=20)
            plt.xlabel('Age', fontsize=18)
            plt.ylabel(col.replace("_"," ").title(), fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2)
            plt.show()
            '''
    return scattered_json


def binning(df,col,num_bins,labels=[]):
    '''
    binning for age using histogram analysis and creates a new column with bin labels

    optional labels parameter - if there is a defined definition for the label
    '''
    print(len(labels))
    hist,bin_edges = np.histogram(df[col],num_bins)
    #filter
    start = bin_edges[0]
    bins = list()
    plt.figure(figsize=(10,7))
    
    #add bin to column
    df[str(col)+'_bin'] = None
    for i in range(1,len(bin_edges)):
        
        #labelling
        if len(labels) != num_bins:
            if bin_edges[-1] == bin_edges[i]:
                df.loc[(df[col] >= start) & (df[col] <= round(bin_edges[i])),[str(col)+'_bin']]= str(int(round(start)))+ "-" + str(int(round(bin_edges[i])))
            else:
                df.loc[(df[col] >= start) & (df[col] < round(bin_edges[i])),[str(col)+'_bin']]= str(int(round(start)))+ "-" + str(int(round(bin_edges[i])-1))
        else:
            if bin_edges[-1] == bin_edges[i]:
                df.loc[(df[col] >= start) & (df[col] <= round(bin_edges[i])),[str(col)+'_bin']]= labels[i-1]
            else:
                df.loc[(df[col] >= start) & (df[col] < round(bin_edges[i])),[str(col)+'_bin']]= labels[i-1]
        start = round(bin_edges[i])
    return df



  
#print(scattered_json())            
'''
chosen_col = ['chest_pain','electrocardiographic_result']#,  'electrocardiographic_result', 'major_vessels', 'thal', 'target']

df = get_data("processed.cleveland.data")
#example bin size is 8
labels=['a','b','c','d','e','f','g','h']
df = binning(df,'age',8)
age_stacked_bar(chosen_col,df)
#age_scattered(df)
'''

