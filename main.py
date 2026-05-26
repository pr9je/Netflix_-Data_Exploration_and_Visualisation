# importing libraries for our purpose.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
df = pd.read_csv('/content/netflix.csv')
df.head()

# length of data
print(len(df))

# checking datatypes
df.dtypes

# number of unique values in our data
for i in df.columns:
  print(i,':',df[i].nunique())

# checking null values in every column of dataset.
df.isnull().sum()

# unnesting the directors column, i.e- creating separate lines for each director in moive 
constraint1 = df['director'].apply(lambda x: str(x).split(', ')).tolist()
df_new1 = pd.DataFrame(constraint1, index=df['title'])
df_new1 = df_new1.stack()
df_new1 = pd.DataFrame(df_new1.reset_index())
df_new1.rename(columns={0: 'Directors'},inplace=True)
df_new1.drop(['level_1'],axis=1,inplace=True)
df_new1