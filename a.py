import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def get_data(file):
    usecols=['age','sex','chest_pain','blood_pressure','serum_cholesterol','blood_sugar','electrocardiographic_result','max_heart_rate','exercise_induced','old_peak','slope','major_vessels','thal','target']
    df = pd.read_csv(file,sep=',',header=None ,names=usecols)
    return df

def age_scattered(df):
    
    #plt.show()
    for col in df:
        if col != 'age' and col != 'sex':
            '''
            plt.scatter(x=df['age'],y=df[col],c=df['sex'])

            plt.show()
            '''
            male = df[df['sex'] == 1]
            female = df[df['sex'] == 0]
            plt.scatter(x=male['age'],y=male[col],c='Blue', label='Male')
            plt.scatter(x=female['age'],y=female[col],c='Pink', label='Female')
            #plt.title('Age (x-axis) and ' + col.replace("_"," ").title()+' (y-axis)', fontsize=20)
            plt.xlabel('Age', fontsize=18)
            plt.ylabel(col.replace("_"," ").title(), fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2)
            plt.show()
    
df = get_data("processed.cleveland.data")
age_scattered(df)


