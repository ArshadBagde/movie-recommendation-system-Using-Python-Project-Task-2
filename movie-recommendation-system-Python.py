#!/usr/bin/env python
# coding: utf-8

# #  Project:- movie-recommendation-system-Using-Python Project
# Organization:- Infopillar Solution

# # Data Science Internship
# Author:- Arshad R. Bagde

# In[ ]:





# # Importing libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ## Loading data files

# In[2]:


movies=pd.read_csv('../input/movies.csv')
ratings=pd.read_csv('../input/ratings.csv')


# In[3]:


movies.info()


# In[4]:


ratings.info()


# In[5]:


movies.shape


# In[6]:


ratings.shape


# In[7]:


movies.describe()


# In[8]:


ratings.describe()


# In[9]:


genres=[]
for genre in movies.genres:
    
    x=genre.split('|')
    for i in x:
         if i not in genres:
            genres.append(str(i))
genres=str(genres)    
movie_title=[]
for title in movies.title:
    movie_title.append(title[0:-7])
movie_title=str(movie_title)    


# ## Data Visualization

# In[10]:


wordcloud_genre=WordCloud(width=1500,height=800,background_color='black',min_font_size=2
                    ,min_word_length=3).generate(genres)
wordcloud_title=WordCloud(width=1500,height=800,background_color='cyan',min_font_size=2
                    ,min_word_length=3).generate(movie_title)


# In[11]:


plt.figure(figsize=(30,10))
plt.axis('off')
plt.title('WORDCLOUD for Movies Genre',fontsize=30)
plt.imshow(wordcloud_genre)


# In[12]:


plt.figure(figsize=(30,10))
plt.axis('off')
plt.title('WORDCLOUD for Movies title',fontsize=30)
plt.imshow(wordcloud_title)


# In[13]:


df=pd.merge(ratings,movies, how='left',on='movieId')
df.head()


# In[14]:


df1=df.groupby(['title'])[['rating']].sum()
high_rated=df1.nlargest(20,'rating')
high_rated.head()


# In[15]:


plt.figure(figsize=(30,10))
plt.title('Top 20 movies with highest rating',fontsize=40)
colors=['red','yellow','orange','green','magenta','cyan','blue','lightgreen','skyblue','purple']
plt.ylabel('ratings',fontsize=30)
plt.xticks(fontsize=25,rotation=90)
plt.xlabel('movies title',fontsize=30)
plt.yticks(fontsize=25)
plt.bar(high_rated.index,high_rated['rating'],linewidth=3,edgecolor='red',color=colors)


# In[16]:


df2=df.groupby('title')[['rating']].count()
rating_count_20=df2.nlargest(20,'rating')
rating_count_20.head()


# In[17]:


plt.figure(figsize=(30,10))
plt.title('Top 20 movies with highest number of ratings',fontsize=30)
plt.xticks(fontsize=25,rotation=90)
plt.yticks(fontsize=25)
plt.xlabel('movies title',fontsize=30)
plt.ylabel('ratings',fontsize=30)

plt.bar(rating_count_20.index,rating_count_20.rating,color='red')


# In[18]:


cv=TfidfVectorizer()
tfidf_matrix=cv.fit_transform(movies['genres'])


# In[19]:


movie_user = df.pivot_table(index='userId',columns='title',values='rating')
movie_user.head()


# In[20]:



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[21]:


indices=pd.Series(movies.index,index=movies['title'])
titles=movies['title']
def recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[22]:


recommendations('Toy Story (1995)')

