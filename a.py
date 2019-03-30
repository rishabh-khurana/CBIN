import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from io import StringIO

def get_data(file):
    usecols=['age','sex','chest_pain','blood_pressure','serum_cholesterol','blood_sugar','electrocardiographic_result','max_heart_rate','exercise_induced','old_peak','slope','major_vessels','thal','target']
    df = pd.read_table(file,sep=',',header=None ,names=usecols)
    return df



df = get_data("processed.cleveland.data")

print(df.head())
