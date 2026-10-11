# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')

# Global plot style
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'figure.facecolor': 'white'})

df = pd.read_csv('/content/netflix.csv')
df.head()

# Basic Metrics 
total     = len(df)
movies    = (df['type'] == 'Movie').sum()
tv        = (df['type'] == 'TV Show').sum()
countries = df['country'].dropna().str.split(', ').explode().nunique()
directors = df['director'].dropna().str.split(', ').explode().nunique()
actors    = df['cast'].dropna().str.split(', ').explode().nunique()
genres    = df['listed_in'].str.split(', ').explode().nunique()

print('=' * 52)
print('        N E T F L I X   D A T A   S U M M A R Y')
print('=' * 52)
print(f'  Total Titles         : {total:,}')
print(f'  Movies               : {movies:,}  ({movies/total*100:.1f}%)')
print(f'  TV Shows             : {tv:,}  ({tv/total*100:.1f}%)')
print(f'  Unique Countries     : {countries}')
print(f'  Unique Directors     : {directors:,}')
print(f'  Unique Actors        : {actors:,}')
print(f'  Unique Genres        : {genres}')
print(f'  Release Year Range   : {int(df.release_year.min())} – {int(df.release_year.max())}')
print('=' * 52)

# Shape, Datatypes, Categorical Conversion, Missing Values & Statistical Summary
# Shape of data
print(f'Shape: {df.shape} -> {df.shape[0]:,} rows, {df.shape[1]} columns')
df.head()

# Data Types (before conversion)
print('Data types (original):')
print(df.dtypes)
print()
df.info()

# Convert categorical columns to 'category' dtype 
# 'type' and 'rating' have low cardinality — converting reduces memory usage
# and makes category-aware operations more efficient.
df['type']   = df['type'].astype('category')
df['rating'] = df['rating'].astype('category')

print('Columns converted to category dtype:')
print(df[['type','rating']].dtypes)
print()
print(f"  'type'   categories : {df['type'].cat.categories.tolist()}")
print(f"  'rating' categories : {sorted(df['rating'].dropna().cat.categories.tolist())}")

# Missing Value Detection 
missing = pd.DataFrame({
    'Missing Count' : df.isnull().sum(),
    'Missing %'     : (df.isnull().sum() / len(df) * 100).round(2)
}).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

print('Missing Values Summary:')
print(missing.to_string())

# Missing value bar chart
fig, ax = plt.subplots(figsize=(9, 4))
missing['Missing %'].plot(kind='barh', ax=ax, color='salmon', edgecolor='black')
ax.set_xlabel('Missing %', fontsize=12)
ax.set_title('Missing Value Percentage by column', fontsize=14, fontweight='bold')
for bar in ax.patches:
  ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
          f'{bar.get_width():.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.show()

# Non-graphical Analysis: Value count & Unique Attributes
# length of data
len(df)

# Checking datatypes
df.dtypes

# number of unique values in our data
for i in df.columns:
  print(i,':', df[i].nunique)

# Checking null values in every column of our data
df.isnull().sum()

# Checking the occurences of each of the ratings.
df['rating'].value_counts()

## Pre-Processing: Unnesting Multi-Value Columns

con1 = df['director'].apply(lambda x: str(x).split(', ')).tolist()
con1[:3]

df_new1 = pd.DataFrame(con1,index=df['title'])
df_new1.head()

constraint1 = df['director'].apply(lambda x: str(x).split(', ')).tolist()
df_new1 = pd.DataFrame(constraint1, index=df['title'])
df_new1 = df_new1.stack()
df_new1 = pd.DataFrame(df_new1.reset_index())
df_new1.sample(5)

# unnesting directors column, i.e. creating separate lines for each director in a moive. 
constraint1 = df['director'].apply(lambda x: str(x).split(', ')).tolist()
df_new1 = pd.DataFrame(constraint1, index=df['title'])
df_new1 = df_new1.stack()
df_new1 = pd.DataFrame(df_new1.reset_index())
df_new1.rename(columns={0:'Directors'}, inplace=True)
df_new1.drop(['level_1'], axis=1,inplace=True)
df_new1.head()

# unnesting the cast column, i.e. - creating separate lines for each cast member in movie
constraint2 = df['cast'].apply(lambda x: str(x).split(', ')).tolist()
df_new2 = pd.DataFrame(constraint2, index=df['title'])
df_new2=df_new2.stack()
df_new2=pd.DataFrame(df_new2.reset_index())
df_new2.rename(columns={0:'Cast'}, inplace=True)
df_new2.drop(['level_1'], axis=1, inplace=True)
df_new2.head()

# unnesting the listed_in column, i.e. creating separate lines for each genre in a movie
constraint3 = df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df_new3 = pd.DataFrame(constraint3, index=df['title'])
df_new3=df_new3.stack()
df_new3= pd.DataFrame(df_new3.reset_index())
df_new3.rename(columns={0:'Genre'},inplace=True)
df_new3.drop(['level_1'], axis=1, inplace=True)
df_new3.head()

# unnesting the country column, i.e. creating separate lines for each country in a movie
constraint4 = df['country'].apply(lambda x: str(x).split(', ')).tolist()
df_new4 = pd.DataFrame(constraint4,index=df['title'])
df_new4 = df_new4.stack()
df_new4 = pd.DataFrame(df_new4.reset_index())
df_new4.rename(columns={0: 'country'},inplace=True)
df_new4.drop(['level_1'], axis=1, inplace=True)
df_new4.head()

#merging the unnested director data with unnested actors data
df_new5 = df_new2.merge(df_new1, on=['title'],how='inner')

#merging the above merged data with unnested genre data
df_new6 = df_new5.merge(df_new3, on=['title'],how='inner')

# merging the above merged data with unnested country data
df_new = df_new6.merge(df_new4, on=['title'], how='inner')

# replacing nan values od directore and actor by unknown actor and director
df_new['Actors'].replace(['nan'],['Unknown Actor'], inplace=True)
df_new['Directors'].replace(['nan'],['Unknown Director'],inplace=True)
df_new['country'].replace(['nan'], [np.nan],inplace=True)
df_new.head(20)

#merging our unnested data with the original data
df_final=df_new.merge(df[['show_id', 'type', 'title', 'date_added',
       'release_year', 'rating', 'duration']],on=['title'],how='left')
df_final.head()
