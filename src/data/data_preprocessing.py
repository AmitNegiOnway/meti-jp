import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml

#  Load test_size from params.yaml
with open('params.yaml')as f:
    params=yaml.safe_load(f)
test_size=params['data_preprocessing']['test_size']

df=pd.read_csv('data/raw/LungCapData.csv')
df.drop("Unnamed: 0",axis=1 ,inplace=True)
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(df,test_size=0.2,random_state=1)

# store the data inside data/processed
data_path=os.path.join('data','processed')
os.makedirs(data_path,exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
train_data.to_csv(os.path.join(data_path,'train_data.csv'),index=False)
test_data.to_csv(os.path.join(data_path,'test_data.csv'),index=False)






