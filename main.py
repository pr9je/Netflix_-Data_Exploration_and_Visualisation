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

# number of distinct titles on the basis of duration.
df_final1.groupby(['duration']).agg({"title":"nunique"})

df_duration = df_final1.groupby(['duration']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15, 8))
bar = plt.barh(df_duration[::-1]['duration'], df_duration[::-1]['title'],color=['pink'])
plt.xlabel('Frequency by Duration')
plt.ylabel('Duration')
plt.title('Top Duration Buckets on Netflix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── HISTOGRAM: Movie Duration (numeric, minutes) ─────────────────────────────
movies_num = df_final.copy()
movies_num = movies_num[movies_num['type']=='Movie'].copy()
movies_num['dur_num'] = pd.to_numeric(
    movies_num['duration'].str.replace(' min','').str.replace('min',''), errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(movies_num['dur_num'].dropna(), bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Movie Duration — Histogram', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Duration (minutes)')
axes[0].set_ylabel('Count')

# KDE / Distplot replacement
sns.kdeplot(movies_num['dur_num'].dropna(), ax=axes[1],
            fill=True, color='steelblue', alpha=0.6, linewidth=2.5)
axes[1].set_title('Movie Duration — KDE (Distplot)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Duration (minutes)')

plt.suptitle('Univariate: Continuous Variable — Movie Duration', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

print(f'\nMedian movie duration: {movies_num["dur_num"].median():.0f} min  |  '
      f'Mean: {movies_num["dur_num"].mean():.0f} min  |  '
      f'Max: {movies_num["dur_num"].max():.0f} min')

# number of distinct titles on the basis of actors
df_final1.groupby(['Actors']).agg({"title":"nunique"})

df_actors=df_final1.groupby(['Actors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:31]
df_actors=df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,8))
bars = plt.barh(df_actors[::-1]['Actors'], df_actors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies')
plt.ylabel('Popular Actors')
plt.title('Top Actors by Title Count (Overall)', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

#number of distinct titles on the basis of Directors
df_final1.groupby(['Directors']).agg({"title":"nunique"})

df_directors=df_final1.groupby(['Directors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:31]
df_directors=df_directors[df_directors['Directors']!='Unknown Director']
plt.figure(figsize=(15,8))
bars = plt.barh(df_directors[::-1]['Directors'], df_directors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies')
plt.ylabel('Popular Directors')
plt.title('Top Directors by Title Count (Overall)', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# number of distinct titles on the basis of year.
df_final1.groupby(['year']).agg({"title":"nunique"})

df_year=df_final1.groupby(['year']).agg({"title":"nunique"}).reset_index()
plt.figure(figsize=(13,5))
ax = sns.lineplot(data=df_year, x='year', y='title', marker='o', linewidth=2.5, color='royalblue')

# Explicitly annotating only the distinct title count (y-value) on each data point
for x, y in zip(df_year['year'], df_year['title']):
    plt.text(x, y + 40, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')

plt.ylabel("Titles Added in the Year")
plt.xlabel("Year")
plt.title("Content Added to Netflix Per Year", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# number of distinct titles on the basis of week.
df_final1.groupby(['week_Added']).agg({"title":"nunique"})


df_week=df_final1.groupby(['week_Added']).agg({"title":"nunique"}).reset_index()
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=df_week, x='week_Added', y='title', linewidth=2, color='darkorange', marker='o')

# Adding y-values (title count) to each point on the line chart
for x, y in zip(df_week['week_Added'], df_week['title']):
    plt.text(x, y + 8, f'{int(y)}', ha='center', va='bottom', fontsize=8, color='black', fontweight='semibold')

plt.ylabel("Titles Released in the Week")
plt.xlabel("Week No.")
plt.title('Content Addition by Week of Year', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

df_release_year=df_final1[df_final1['release_year']>=1980].groupby(['release_year']).agg({"title":"nunique"}).reset_index()
plt.figure(figsize=(13,5))
ax = sns.lineplot(data=df_release_year, x='release_year', y='title', linewidth=2.5, color='crimson')

# Finding the minimum and maximum data points
min_row = df_release_year.loc[df_release_year['title'].idxmin()]
max_row = df_release_year.loc[df_release_year['title'].idxmax()]

# Annotating the minimum point
plt.text(min_row['release_year'], min_row['title'] + 30, f"Min: {int(min_row['title'])}", 
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

# Annotating the maximum point
plt.text(max_row['release_year'], max_row['title'] + 30, f"Max: {int(max_row['title'])}", 
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

plt.ylabel("Titles by Actual Release Year")
plt.xlabel("Year")
plt.title('Actual Release Year of Netflix Content (from 1980)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

### Categorical Variables - Boxplots & Bivariate Analysis
## Univariate Analysis separatedly for TV Shows and Movies
df_shows = df_final1[df_final1['type'] == 'TV Show']
df_movies = df_final1[df_final1['type'] == 'Movie']

# Genre - TV Shows
df_genre = df_shows.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_genre[::-1]['Genre'], df_genre[::-1]['title'], color=['orange'])
plt.xlabel('Frequency of Genres')
plt.ylabel("Genres")
plt.title('Top 15 Genres - TV Shows', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()


# Genre -Movies
df_genre = df_movies.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'], ascending=False)[:15]
plt.figure(figsize=(15, 8))
bar = plt.barh(df_genre[::-1]['Genre'],df_genre[::-1]['title'],color=['orange'])
plt.xlabel('Frequency of Genres')
plt.ylabel("Genres")
plt.title('Top 15 Genres - Movies', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bar, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

df_country=df_shows.groupby(['country']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,8))
bars = plt.barh(df_country[::-1]['country'], df_country[::-1]['title'],color=['blue'])
plt.xlabel('Titles by Countries')
plt.ylabel('Countries')
plt.title('Top 10 Content-Producing Countries - TV Shows', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Country - Movies
df_country=df_movies.groupby(['country']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,8))
bars = plt.barh(df_country[::-1]['country'], df_country[::-1]['title'],color=['blue'])
plt.xlabel('Titles by Countries')
plt.ylabel('Countries')
plt.title('Top 10 Content-Producing Countries - Movies', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Rating — TV Shows
df_rating=df_shows.groupby(['rating']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_rating[::-1]['rating'], df_rating[::-1]['title'],color=['violet'])
plt.xlabel('Frequency by Ratings')
plt.ylabel('Ratings')
plt.title('Rating Distribution — TV Shows', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Rating — Movies
df_rating=df_movies.groupby(['rating']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_rating[::-1]['rating'], df_rating[::-1]['title'],color=['violet'])
plt.xlabel('Frequency by Ratings')
plt.ylabel('Ratings')
plt.title('Rating Distribution — Movies', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Duration - TV- Shows 
df_duration = df_shows.groupby(['duration']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,8))
bars = plt.barh(df_duration[::-1]['duration'], df_duration[::-1]['title'],color=['pink'])
plt.xlabel('Frequency by Duration')
plt.ylabel('Duration')
plt.title('Duration Distribution — TV Shows (Seasons)', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
# Duration — Movies
df_duration=df_movies.groupby(['duration']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,8))
bars = plt.barh(df_duration[::-1]['duration'], df_duration[::-1]['title'],color=['pink'])
plt.xlabel('Frequency by Duration')
plt.ylabel('Duration')
plt.title('Duration Distribution — Movies (Minute Buckets)', fontsize=14, fontweight='bold')

# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Acotos - TV Shows
df_actors = df_shows.groupby(['Actors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:31]
df_actors = df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,8))
bars = plt.barh(df_actors[::-1]['Actors'], df_actors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies')
plt.ylabel('Popular Actors')
plt.title('Top Actors by Title Count - TV Shows',fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Acotos - TV Shows
df_actors = df_movies.groupby(['Actors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:31]
df_actors = df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,8))
bars = plt.barh(df_actors[::-1]['Actors'], df_actors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies')
plt.ylabel('Popular Actors')
plt.title('Top Actors by Title Count - Movies',fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Directors — TV Shows
df_directors=df_final1.groupby(['Directors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:31]
df_directors=df_directors[df_directors['Directors']!='Unknown Director']
plt.figure(figsize=(15,8))
bars = plt.barh(df_directors[::-1]['Directors'], df_directors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies')
plt.ylabel('Popular Directors')
plt.title('Top Directors by Title Count (Overall)', fontsize=14, fontweight='bold')
# Adding values to the individual bars
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# number of distinct titles on the basis of year
df_final1.groupby(['year']).agg({"title":"nunique"})

df_year= df_final1.groupby(['year']).agg({"title":"nunique"}).reset_index()
plt.figure(figsize=(13,5))
ax = sns.lineplot(data=df_year, x='year', y='title', marker='o',linewidth=2.5, color='royalblue')

# Adding values to each data point
for x, y in zip(df_year['year'], df_year['title']):
    plt.text(x, y + 40, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')

plt.ylabel("Titles Added in the Year")
plt.xlabel("Year")
plt.title('Content Added to Netflix Per Year', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Year trend - TV Shows vs Movies (Bivariate Line)

df_year_shows = df_shows.groupby(['year']).agg({"title":"nunique"}).reset_index()
df_year_movies = df_movies.groupby(['year']).agg({"title":"nunique"}).reset_index()

fig, axes = plt.subplots(1,2,figsize=(16,5))

# TV Shows
sns.lineplot(data=df_year_shows,  x='year', y='title', ax=axes[0], marker='o', color='steelblue', linewidth=2)
axes[0].set_ylabel("Shows Added in the Year"); axes[0].set_xlabel("Year")
axes[0].set_title("TV Shows Added Per Year", fontweight='bold')
for x, y in zip(df_year_shows['year'], df_year_shows['title']):
    axes[0].text(x, y + 15, f'{int(y)}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkblue')

# Movies
sns.lineplot(data=df_year_movies, x='year', y='title', ax=axes[1], marker='o', color='coral',    linewidth=2)
axes[1].set_ylabel("Movies Added in the Year"); axes[1].set_xlabel("Year")
axes[1].set_title("Movies Added Per Year", fontweight='bold')
for x, y in zip(df_year_movies['year'], df_year_movies['title']):
    axes[1].text(x, y + 30, f'{int(y)}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='brown')

plt.tight_layout()
plt.show()

# Week trend — TV Shows vs Movies
df_week_shows =df_shows.groupby(['week_Added']).agg({"title":"nunique"}).reset_index()
df_week_movies=df_movies.groupby(['week_Added']).agg({"title":"nunique"}).reset_index()

fig, axes = plt.subplots(1,2,figsize=(16,5))

# TV Shows
sns.lineplot(data=df_week_shows,  x='week_Added', y='title', ax=axes[0], color='steelblue', marker='o', linewidth=2)
axes[0].set_ylabel("Shows Released in the Week"); axes[0].set_xlabel("Week No.")
axes[0].set_title("TV Shows Added by Week", fontweight='bold')
for x, y in zip(df_week_shows['week_Added'], df_week_shows['title']):
    axes[0].text(x, y + 2, f'{int(y)}', ha='center', va='bottom', fontsize=7, color='black')

# Movies
sns.lineplot(data=df_week_movies, x='week_Added', y='title', ax=axes[1], color='coral', marker='o', linewidth=2)
axes[1].set_ylabel("Movies Released in the Week"); axes[1].set_xlabel("Week No.")
axes[1].set_title("Movies Added by Week", fontweight='bold')
for x, y in zip(df_week_movies['week_Added'], df_week_movies['title']):
    axes[1].text(x, y + 5, f'{int(y)}', ha='center', va='bottom', fontsize=7, color='black')

plt.tight_layout()
plt.show()

# Month trend — TV Shows vs Movies
df_month_shows =df_shows.groupby(['month_added']).agg({"title":"nunique"}).reset_index()
df_month_movies=df_movies.groupby(['month_added']).agg({"title":"nunique"}).reset_index()

fig, axes = plt.subplots(1,2,figsize=(16,5))

# TV Shows
sns.lineplot(data=df_month_shows,  x='month_added', y='title', ax=axes[0], marker='o', color='steelblue', linewidth=2)
axes[0].set_ylabel("TV Shows Released in the Month"); axes[0].set_xlabel("Month")
axes[0].set_title("TV Shows Added by Month", fontweight='bold')
for x, y in zip(df_month_shows['month_added'], df_month_shows['title']):
    axes[0].text(x, y + 8, f'{int(y)}', ha='center', va='bottom', fontsize=8, color='black', fontweight='semibold')

# Movies
sns.lineplot(data=df_month_movies, x='month_added', y='title', ax=axes[1], marker='o', color='coral',    linewidth=2)
axes[1].set_ylabel("Movies Released in the Month"); axes[1].set_xlabel("Month")
axes[1].set_title("Movies Added by Month", fontweight='bold')
for x, y in zip(df_month_movies['month_added'], df_month_movies['title']):
    axes[1].text(x, y + 15, f'{int(y)}', ha='center', va='bottom', fontsize=8, color='black', fontweight='semibold')

plt.tight_layout()
plt.show()

# Release Year trend - Shows vs Movies
df_ry_shows = df_shows[df_shows['release_year']>=1980].groupby(['release_year']).agg({"title":"nunique"}).reset_index()
df_ry_movies = df_movies[df_movies['release_year']>=1980].groupby(['release_year']).agg({"title":"nunique"}).reset_index()

fig, axes = plt.subplots(1,2,figsize=(16,5))

# TV Shows
sns.lineplot(data=df_ry_shows,  x='release_year', y='title', ax=axes[0], color='steelblue', linewidth=2)
axes[0].set_ylabel("TV Shows Actual Release Year")
axes[0].set_xlabel("Year")
axes[0].set_title("TV Shows — Actual Release Year (≥1980)", fontweight='bold')

# Annotate Min and Max for TV Shows
ry_shows_min = df_ry_shows.loc[df_ry_shows['title'].idxmin()]
ry_shows_max = df_ry_shows.loc[df_ry_shows['title'].idxmax()]
axes[0].text(ry_shows_min['release_year'], ry_shows_min['title'] + 10, f"Min: {int(ry_shows_min['title'])}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
axes[0].text(ry_shows_max['release_year'], ry_shows_max['title'] + 10, f"Max: {int(ry_shows_max['title'])}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Movies
sns.lineplot(data=df_ry_movies, x='release_year', y='title', ax=axes[1], color='coral',    linewidth=2)
axes[1].set_ylabel("Movies Actual Release Year")
axes[1].set_xlabel("Year")
axes[1].set_title("Movies — Actual Release Year (≥1980)", fontweight='bold')

# Annotate Min and Max for Movies
ry_movies_min = df_ry_movies.loc[df_ry_movies['title'].idxmin()]
ry_movies_max = df_ry_movies.loc[df_ry_movies['title'].idxmax()]
axes[1].text(ry_movies_min['release_year'], ry_movies_min['title'] + 25, f"Min: {int(ry_movies_min['title'])}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
axes[1].text(ry_movies_max['release_year'], ry_movies_max['title'] + 25, f"Max: {int(ry_movies_max['title'])}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()

# ── BOXPLOT 1: Movie Duration (minutes) by Rating ─────────────────────────────
top_ratings = movies_num['rating'].value_counts().head(8).index
box_data = movies_num[movies_num['rating'].isin(top_ratings)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=box_data, x='rating', y='dur_num', order=top_ratings,
            ax=ax, palette='Set3')
ax.set_title('Boxplot: Movie Duration by Content Rating', fontsize=14, fontweight='bold')
ax.set_xlabel('Content Rating')
ax.set_ylabel('Duration (minutes)')
plt.tight_layout()
plt.show()

# ── BOXPLOT 2: Release Year by Content Type ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df_final1, x='type', y='release_year',
            ax=ax, palette=['#4C72B0','#DD8452'])
ax.set_title('Boxplot: Release Year by Content Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Content Type')
ax.set_ylabel('Release Year')
plt.tight_layout()
plt.show()


# ── BOXPLOT 3: Duration (numeric) by Content Type ────────────────────────────
# Extract numeric values for both movies (minutes) and TV shows (seasons)
# We fetch the clean original TV show durations from 'df' to avoid the binned values in 'df_final1'
tv_shows_orig = df[df['type'] == 'TV Show'].copy()
tv_shows_orig['seasons'] = tv_shows_orig['duration'].str.extract(r'(\d+)').astype(float)

movies_num2 = movies_num[['type','dur_num']].rename(columns={'dur_num':'value'})
df_tv_plot  = tv_shows_orig[['type','seasons']].rename(columns={'seasons':'value'})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: Movie Durations (minutes)
sns.boxplot(data=movies_num2, x='type', y='value', ax=axes[0], color='#4C72B0', width=0.4)
axes[0].set_title('Boxplot: Movie Duration', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Content Type')
axes[0].set_ylabel('Duration (minutes)')

# Right subplot: TV Show Durations (seasons)
sns.boxplot(data=df_tv_plot, x='type', y='value', ax=axes[1], color='#DD8452', width=0.4)
axes[1].set_title('Boxplot: TV Show Seasons', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Content Type')
axes[1].set_ylabel('Number of Seasons')

plt.suptitle('Comparison of Durations (Movies vs TV Shows)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()



# Filter for United States once to improve performance and readability
df_usa = df_final1[df_final1['country'] == 'United States']

# Split into TV Shows and Movies
df_usa_shows = df_usa[df_usa['type'] == 'TV Show']
df_usa_movies = df_usa[df_usa['type'] == 'Movie']

df_genre = df_usa_shows.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_genre[::-1]['Genre'], df_genre[::-1]['title'], color=['orange'])
plt.xlabel('Frequency of Genres')
plt.ylabel("Genres")
plt.title('Top Genres - USA TV Shows', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout();plt.show()

df_genre = df_usa_movies.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'], ascending=False)[:15]
plt.figure(figsize=(15, 8))
bar = plt.barh(df_genre[::-1]['Genre'],df_genre[::-1]['title'],color=['orange'])
plt.xlabel('Frequency of Genres')
plt.ylabel("Genres")
plt.title('Top Genres - USA Movies', fontsize=14, fontweight='bold')
plt.bar_label(bar, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout();plt.show()

df_rating=df_usa_shows.groupby(['rating']).agg({"title":"nunique"}).reset_index().sort_values(by=["title"],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_rating[::-1]['rating'], df_rating[::-1]['title'],color=['violet'])
plt.xlabel('Frequency by Ratings'); plt.ylabel('Ratings')
plt.title('Rating Distribution — USA TV Shows', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_rating=df_usa_movies.groupby(['rating']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(15,8))
bars = plt.barh(df_rating[::-1]['rating'], df_rating[::-1]['title'],color=['violet'])
plt.xlabel('Frequency by Ratings'); plt.ylabel('Ratings')
plt.title('Rating Distribution — USA Movies', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_duration=df_usa_movies.groupby(['duration']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,8))
bars = plt.barh(df_duration[::-1]['duration'], df_duration[::-1]['title'],color=['pink'])
plt.xlabel('Frequency by Duration'); plt.ylabel('Duration')
plt.title('Movie Duration — USA', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_actors=df_usa_shows.groupby(['Actors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
df_actors=df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,8))
bars = plt.barh(df_actors[::-1]['Actors'], df_actors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Shows'); plt.ylabel('Popular Actors')
plt.title('Top Actors — USA TV Shows', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_actors=df_usa_movies.groupby(['Actors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
df_actors=df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,8))
bars = plt.barh(df_actors[::-1]['Actors'], df_actors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies'); plt.ylabel('Popular Actors')
plt.title('Top Actors — USA Movies', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_directors=df_usa_shows.groupby(['Directors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
df_directors=df_directors[df_directors['Directors']!='Unknown Director']
plt.figure(figsize=(15,8))
bars = plt.barh(df_directors[::-1]['Directors'], df_directors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Shows'); plt.ylabel('Popular Directors')
plt.title('Top Directors — USA TV Shows', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()

df_directors=df_usa_movies.groupby(['Directors']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
df_directors=df_directors[df_directors['Directors']!='Unknown Director']
plt.figure(figsize=(15,8))
bars = plt.barh(df_directors[::-1]['Directors'], df_directors[::-1]['title'],color=['indigo'])
plt.xlabel('Number of Movies'); plt.ylabel('Popular Directors')
plt.title('Top Directors — USA Movies', fontsize=14, fontweight='bold')
plt.bar_label(bars, padding=5, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()



df_year=df_usa_shows.groupby(['year']).agg({"title":"nunique"}).reset_index()
sns.lineplot(data=df_year, x='year', y='title', marker='o', linewidth=2)
plt.ylabel("Shows Released in the Year"); plt.xlabel("Year")
plt.title("USA TV Shows Added Per Year", fontweight='bold') 
for x, y in zip(df_year['year'], df_year['title']):
  plt.text(x, y + 15, f'{int(y)}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkblue')
plt.tight_layout(); plt.show()


import plotly.express as px
import plotly.io as pio

# Set the default renderer for Google Colab
pio.renderers.default = 'colab'

# Group the data to count unique movies per year
df_year = df_usa_movies.groupby(['year']).agg({"title":"nunique"}).reset_index()

# Plot the data using Plotly Express
fig = px.line(
    df_year, 
    x='year', 
    y='title', 
    markers=True, 
    title='USA Movies Added Per Year',
    labels={'year': 'Year', 'title': 'Movies Released in the Year'}
)

# Customize layout aesthetics
fig.update_layout(
    title_font=dict(size=16, family="Arial", color="black"),
    title_x=0.5,
    template='plotly_white'
)

fig.show()

import plotly.express as px
import plotly.io as pio

# Set the default renderer for Google Colab
pio.renderers.default = 'colab'

# Group the data to count unique shows per week
df_week = df_usa_shows.groupby(['week_Added']).agg({"title":"nunique"}).reset_index()

# Plot the data using Plotly Express
fig = px.line(
    df_week,
    x='week_Added',
    y='title',
    markers=True,
    title='USA TV Shows — Week Added',
    labels={'week_Added': 'Week No.', 'title': 'Shows Released in the Week'}
)

# Customize layout aesthetics
fig.update_layout(
    title_font=dict(size=16, family="Arial", color="black"),
    title_x=0.5,
    template='plotly_white'
)

fig.show()

import plotly.express as px
import plotly.io as pio

# Set the default renderer for Google Colab
pio.renderers.default = 'colab'

# Group the data to count unique shows per week
df_week = df_usa_movies.groupby(['week_Added']).agg({"title":"nunique"}).reset_index()

# Plot the data using Plotly Express
fig = px.line(
    df_week,
    x='week_Added',
    y='title',
    markers=True,
    title='USA TV Movies — Week Added',
    labels={'week_Added': 'Week No.', 'title': 'Movies Released in the week'}
)

# Customize layout aesthetics
fig.update_layout(
    title_font=dict(size=16, family="Arial", color="black"),
    title_x=0.5,
    template='plotly_white'
)

fig.show()
