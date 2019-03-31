import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def get_data(file):
    usecols=['age','sex','chest_pain','blood_pressure','serum_cholesterol','blood_sugar','electrocardiographic_result','max_heart_rate','exercise_induced','old_peak','slope','major_vessels','thal','target']
    df = pd.read_csv(file,sep=',',header=None ,names=usecols)
    return df

def age_scattered(df):
    for col in df:
        if col != 'age' and col != 'sex':
            male = df[df['sex'] == 1]
            female = df[df['sex'] == 0]
            plt.scatter(x=male['age'],y=male[col],c='Blue', label='Male')
            plt.scatter(x=female['age'],y=female[col],c='Pink', label='Female')
            #plt.title('Age (x-axis) and ' + col.replace("_"," ").title()+' (y-axis)', fontsize=20)
            plt.xlabel('Age', fontsize=18)
            plt.ylabel(col.replace("_"," ").title(), fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2)
            plt.show()


def binning(df,col,num_bins,labels=[]):
    '''
    binning for age using histogram analysis and creates a new column with bin labels
    '''
    print(len(labels))
    hist,bin_edges = np.histogram(df[col],num_bins)
    #print(hist)
    #print(bin_edges)
    #filter
    start = bin_edges[0]
    bins = list()
    plt.figure(figsize=(10,7))
    #add bin column
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

df = get_data("processed.cleveland.data")
#example bin size is 8
labels=['a','b','c','d','e','f','g','h']
df = binning(df,'age',8,labels)
print(df.head())

#age_scattered(df)


