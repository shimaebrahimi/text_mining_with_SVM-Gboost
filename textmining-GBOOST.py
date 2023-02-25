#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


data=pd.read_csv('per.csv')


# In[3]:


data.head()


# In[4]:


with open('stopwords.txt') as stopwords_file:
    sw=[line.replace('\n','') for line in stopwords_file]


# In[5]:


len(sw)


# In[7]:


nltk.download('stopwords')


# In[8]:


esw=nltk.corpus.stopwords.words('english')


# In[9]:


sw.extend(esw)


# In[10]:


len(sw)


# In[11]:


import hazm


# In[12]:


stemmer=hazm.Stemmer()
lem=hazm.Lemmatizer()


# In[13]:


stemmer.stem('آمده است')
lem.lemmatize('سبزیجات')


# In[ ]:


dataset=pd.DataFrame(columns=['Title_Body','Category'])
for index,Row in data.iterrows():
    title_row=Row['Title']+Row['Body']
    title_row_tokenized=hazm.word_tokenize(title_row)
    title_row_tokenized_filtered=[w for w in title_row_tokenized if w not in sw]
    title_row_tokenized_filtered_stemed=[stemmer.stem(w) for w in title_row_tokenized_filtered]
    title_row_tokenized_filtered_lem=[lem.lemmatize(w).replace('#','') for w in title_row_tokenized_filtered]
    dataset.loc[index]={
        'Title_Body':' '.join(title_row_tokenized_filtered_lem)+' '+' '.join(title_row_tokenized_filtered_stemed),
        'Category':Row['Category2'].replace('\n','')
    }


# In[ ]:


dataset


# In[ ]:


vectorizer=TfidfVectorizer(ngram_range=(1,2))
X=vectorizer.fit_transform(dataset['Title_Body'])


# In[ ]:


X


# In[ ]:


le=LabelEncoder()
y=le.fit_transform(dataset['Category'])


# In[ ]:


y


# In[ ]:


import numpy as np
np.unique(dataset['Category'])


# In[ ]:


np.shape(X)


# In[ ]:


np.shape(y)


# In[ ]:


X


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:


gbc=GradientBoostingClassifier()


# In[ ]:


gbc.fit(x_train,y_train)


# In[ ]:


y_pred=gbc.predict(x_test)


# In[ ]:


print(classification_report(y_test,y_p))


# In[ ]:


print(confusion_matrix(y_test,y_pred))

