#!/usr/bin/env python
# coding: utf-8

# In[98]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, KDD, Knowledge Discovery Data')
os.getcwd()


# In[99]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[100]:


# Importing the numpy and pandas package
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

get_ipython().system('pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')
import pandas_profiling

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


df=pd.read_excel("States.xlsx") 


# In[102]:


df.head()


# In[103]:


df.tail()


# In[104]:


df.astype('object').describe().transpose() 


# In[105]:


df.describe


# In[62]:


df.shape


# In[51]:


df.columns.values


# In[63]:


df.info()


# In[106]:


df.ndim


# In[107]:


df.size   # number of rows*number of columns 


# In[108]:


df.dtypes


# In[109]:


df.apply(lambda x: sum(x.isnull())) # counting missing values


# In[110]:


df.isna().sum()


# In[111]:


ht = sns.heatmap(df.corr(),center=0)


# In[112]:


sns.boxplot(x=df['Ocean'] , y = df['Mort'])


# In[113]:


df1=df


# In[114]:


State = df.State
State_dummies = pd.get_dummies(df.State, prefix='State')

State_dummies.head()


# In[115]:


df.head()


# In[116]:


df1.head()


# In[ ]:





# In[ ]:


X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
X.head()


# In[117]:


df.State.isna().sum()


# In[118]:


#Dummy variable encoding for both Sex and Embarked variable 
df_dummy = pd.get_dummies(df,columns=['State','Mort'],drop_first=True)


# In[119]:


df_dummy.head()


# In[79]:


df.State.value_counts()


# In[90]:


X = pd.DataFrame(dataset.State.xlsx, columns = dataset.feature_names)
X.head()


# In[120]:


df.describe


# In[121]:


df.isna().mean()


# In[122]:




import sklearn 
print (sklearn.__version__)


# In[123]:


import matplotlib.pyplot as plt

# a scatter plot comparing Height and Weight variable
df1.plot(kind='scatter',x='State',y='Mort',color='blue')
plt.show()


# In[124]:


#import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[125]:


#import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[126]:


#train the model
model.fit(X_train,y_train)


# In[93]:


#split the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)


# In[83]:


X = pd.DataFrame(dataset., columns = dataset.feature_names)
X.head()


# In[56]:


plt.scatter(df.State, df.Long)
plt.title('State VS Long')
plt.xlabel('State')
plt.ylabel('Long')
plt.show()


# In[57]:


import sweetviz as sv


# In[58]:


pip install sweetviz


# In[59]:


my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"


# In[ ]:




