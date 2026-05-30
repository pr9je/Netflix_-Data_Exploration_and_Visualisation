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

# unnesting the cast column, i.e., - creating separate lines for each cast member in a moive 
constraint2 = df['cast'].apply(lambda x: str(x).split(', ')).tolist()
df_new2 = pd.DataFrame(constraint2,index=df['title'])
df_new2 = df_new2.stack()
df_new2 = pd.DataFrame(df_new2.reset_index())
df_new2.rename(columns={0: 'Actors'},inplace=True)
df_new2.drop(['level_1'],axis=1,inplace=True)
df_new2.head()

# unnesting the listed_in column, i.e. - creating separate lines for each genre in a movie
constraint3 = df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df_new3 = pd.DataFrame(constraint3,index=df['title'])
df_new3 = df_new3.stack()
df_new3 = pd.DataFrame(df_new3.reset_index())
df_new3.rename(columns={0:'Genre'},inplace=True)
df_new3.drop(['level_1'],axis=1,inplace=True)
df_new3

# unnesting the country column, i.e- creating separate lines for each country in a movie
constraint4 = df['country'].apply(lambda x: str(x).split(', ')).tolist()
df_new4 = pd.DataFrame(constraint4, index=df['title'])
df_new4 = df_new4.stack()
df_new4 = pd.DataFrame(df_new4.reset_index())
df_new4.rename(columns={0: 'country'},inplace=True)
df_new4.drop(['level_1'],axis=1,inplace=True)
df_new4.head()

# mergint the unnested director data with unnested actors data.
df_new5 = df_new2.merge(df_new1, on=['title'],how='inner')

# merging the above merged data with unnested genre data
df_new6 = df_new5.merge(df_new3,on=['title'],how='inner')

# merging the above data with unnested country data.
df_new = df_new6.merge(df_new4,on=['title'],how = 'inner')

# replacing nan values of director and actor by unknown actor and director
df_new['Actors'].replace(['nan'],['Unknown Actor'],inplace=True)
df_new['Directors'].replace(['nan'],['Unknown Director'],inplace=True)
df_new['country'].replace(['nan'],[np.nan],inplace=True)
df_new.head()

# merging our unnested data with the original data
df_final = df_new.merge(df[['show_id', 'type', 'title', 'date_added', 'release_year', 'rating', 'duration']],on=['title'],how='left')
df_final.head()

# checking nulls
df_final.isnull().sum()

# In duration column it was observed that the nulls had values which were written in corresponding rating columns, i.e., you can't expect ratings to be in minutes. So the duration columsns are repladed by corresponding values in raings column.

df_final.loc[df_final['duration'].isnull(),'duration']=df_final.loc[df_final['duration'].isnull(),'duration'].fillna(df_final['rating'])
df_final.loc[df_final['rating'].str.contains('mins',na=False),'rating']='NR'
df_final.isnull().sum()

# Rating can't be in min, so it has been made NR(i.e- Non rated)
df_final.loc[df_final['rating'].str.contains('min',na=False),'rating']='NR'
df_final['rating'].fillna('NR',inplace=True)
pd.set_option('display.max_rows',None)

# nulls in date added column
df_final[df_final['date_added'].isnull()].head()