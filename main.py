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

#now checking nulls
df_final.isnull().sum()

df_final.loc[df_final['duration'].isnull(),'duration']=df_final.loc[df_final['duration'].isnull(),'duration'].fillna(df_final['rating'])
df_final.loc[df_final['rating'].str.contains('min', na=False),'rating']='NR'
df_final.isnull().sum()

# Rating can't ne in minutes, so it has been mad NR(i.e. - Non Rated)
df_final.loc[df_final['rating'].str.contains('min', na=False), 'rating'] = 'NR'
df_final['rating'].fillna('NR', inplace=True)
pd.set_option('display.max_rows',None)

df_final.isnull().sum()

# just an attempt to observe nulls in date_added column
df_final[df_final['date_added'].isnull()].head()

for i in df_final[df_final['date_added'].isnull()]['release_year'].unique():
  imp = df_final[df_final['release_year']==i]['date_added'].mode().values[0]
  df_final.loc[df_final['release_year']==i,'date_added'] = df_final.loc[df_final['release_year']==i,'date_added'].fillna(imp)

#country column is imputed on the basis of director,i.e- suppose there's a null for country
#when we have a director whose other movies have a country given.So below piece of code just checks the mode of
#country for the director
# and imputes in place of nulls the corresponding mode

for i in df_final[df_final['country'].isnull()]['Directors'].unique():
  if i in df_final[~df_final['country'].isnull()]['Directors'].unique():
    imp=df_final[df_final['Directors']==i]['country'].mode().values[0]
    df_final.loc[df_final['Directors']==i,'country']=df_final.loc[df_final['Directors']==i,'country'].fillna(imp)

# Calculate the mode country for each Actor using only non-null country records
actor_modes = df_final[df_final['country'].notnull()].groupby('Actors')['country'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)

# Map the imputed countries onto the missing values
is_null_country = df_final['country'].isnull()
df_final.loc[is_null_country, 'country'] = df_final.loc[is_null_country, 'Actors'].map(actor_modes)

# If there are still nulls, replace them with 'Unknown Country'
df_final['country'].fillna('Unknown Country', inplace=True)

# Output the remaining null counts
df_final.isnull().sum()

df_final['duration'].value_counts()

# removing mins from data
df_final['duration'] = df_final['duration'].str.replace(" min","")
df_final.head()

df_final['duration'].unique()

df_final['duration_copy']=df_final['duration'].copy()
df_final1=df_final.copy()

df_final1.loc[df_final1['duration_copy'].str.contains('Season'),'duration_copy']=0
df_final1['duration_copy']=df_final1['duration_copy'].astype('int')
df_final1.head()

df_final1['duration_copy'].describe()
# Note: sns.distplot is deprecated; using sns.histplot with kde=True (equivalent output)
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_final1['duration_copy'], bins=36, kde=True, color='darkblue',
             edgecolor='black', ax=ax, line_kws={'linewidth': 3})
ax.set_title('Distribution of Duration (numeric — movie minutes, TV shows set to 0)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Duration (minutes) — TV shows shown as 0')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()

bins1 = [-1,1,50,80,100,120,150,200,315]
labels1 = ['<1','1-50','50-80','80-100','100-120','120-150','150-200','200-315']
df_final1['duration_copy'] = pd.cut(df_final1['duration_copy'], bins=bins1, labels=labels1)
df_final1.head()


df_final1.loc[~df_final1['duration'].str.contains('Season'),'duration']=df_final1.loc[~df_final1['duration'].str.contains('Season'),'duration_copy']
df_final1.drop(['duration_copy'],axis=1,inplace=True)
df_final1.head()

df_final1['duration'].value_counts()

from datetime import datetime
from dateutil.parser import parse
arr=[]
for i in df_final1['date_added'].values:
  dt1=parse(i)
  arr.append(dt1.strftime('%Y-%m-%d'))
df_final1['Modified_Added_date'] =arr
df_final1['Modified_Added_date']=pd.to_datetime(df_final1['Modified_Added_date'])
df_final1['month_added']=df_final1['Modified_Added_date'].dt.month
df_final1['week_Added']=df_final1['Modified_Added_date'].dt.isocalendar().week.astype('int')
df_final1['year']=df_final1['Modified_Added_date'].dt.year
df_final1.head()

#Titles such as Bahubali(Hindi Version),Bahubali(Tamil Version) were there. Since it's only one movie in different languages,
#presence of brackets and content between brackets is removed.
df_final1['title']=df_final1['title'].str.replace(r"\(.*\)","",regex=True)
df_final1.head()

# Create Movie and TV Show subsets for split analysis
df_shows=df_final1[df_final1['type']=='TV Show']
df_movies=df_final1[df_final1['type']=='Movie']
print(f'TV Show rows : {len(df_shows):,}')
print(f'Movie rows   : {len(df_movies):,}')

# Visual Analysis - Univariate & Bivariate 
# Continous Variables - Distplot / Histogram / Countplot
# Univariate Analysis in terms of counts of each columns. 
# number of distinct tiles on the basis  of genre
df_final1.groupby(['Genre']).agg({"title":"nunique"})

df_genre = df_final1.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'], ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_genre[::-1]['Genre'], df_genre[::-1]['title'], color=['orange'])
plt.xlabel('Frequency of Genres')
plt.ylabel("Genres")
plt.title('Top 15 Genres on Netflix (Overall)', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# number of distinct titles on the basis of type.
df_type = df_final1.groupby(['type']).agg({"title":"nunique"}).reset_index()
plt.figure(figsize=(7, 7))
plt.pie(df_type['title'], explode=(0.05, 0.05), labels=df_type['type'],
        colors=['purple', 'green'], autopct='%.1f%%',
        wedgeprops=dict(edgecolor='white', linewidth=2))
plt.title('Movies vs TV Shows on Netflix', fontsize=14, fontweight='bold')
plt.show()

# number of distinct titles on the basis of country
df_final.groupby(['country']).agg({"title":"nunique"})

df_final1['country'] = df_final1['country'].str.replace(',','')
df_final1.head()

#number of distinct titles on the basis of country
df_final1.groupby(['country']).agg({"title":"nunique"})

df_country=df_final1.groupby(['country']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:5]
plt.figure(figsize=(15,8))
bars = plt.barh(df_country[::-1]['country'], df_country[::-1]['title'],color=['blue'])
plt.xlabel('Titles by Countries')
plt.ylabel('Countries')
plt.title('Top 5 Content-Producing Countries on Netflix', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# number of distinct titles on the basis of rating.
df_final1.groupby(['rating']).agg({"title":"nunique"})

df_rating=df_final1.groupby(['rating']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_rating[::-1]['rating'], df_rating[::-1]['title'],color=['violet'])
plt.xlabel('Frequency by Ratings')
plt.ylabel('Ratings')
plt.title('Content Rating Distribution on Netflix', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
