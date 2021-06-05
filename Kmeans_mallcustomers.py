#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[17]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, KDD, Knowledge Discovery Data')
os.getcwd()


# In[18]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[19]:


# Importing the numpy and pandas package
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# multi-class classification  we are going to use ROC curve and  AUC score at the end 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[20]:


df=pd.read_csv("Mall_Customers.csv")     #By default keep_default_na=True
#If we don't use na_values='NA' here you won't get missing value for this data  ### ### the source of the data is 
# UCI Machine learning Repository 


# In[22]:


X = df.iloc[:,[-2,-1]]


# In[23]:


plt.scatter(X.iloc[:,0],X.iloc[:,1])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[24]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[25]:


# Training the K-Means model on the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5)
y_kmeans = kmeans.fit_predict(X)


# In[26]:


# Visualising the clusters
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Kmeans for the dataset with more the 2 features

# In[28]:


X = df.iloc[:,1:]


# In[29]:


X.head()


# In[30]:


X.Gender = X.Gender.map({'Male':1, 'Female':0})


# In[31]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[32]:


kmeans = KMeans(n_clusters = 5)
y_kmeans = kmeans.fit_predict(X)


# In[33]:


X.Gender = X.Gender.map({1:'Male', 0:'Female'})


# In[34]:


cluster1 = X.iloc[y_kmeans == 0,:]


# In[35]:


cluster1.describe(include='all')


# In[36]:


cluster2 = X.iloc[y_kmeans == 1,:]
cluster2.describe(include='all')


# In[ ]:





# In[ ]:




