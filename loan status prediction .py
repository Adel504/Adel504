#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
os.chdir(r'C:\Users\Admin\Desktop\python')
os.getcwd()


# In[12]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


# Importing the numpy and pandas package
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

get_ipython().system('pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')
import pandas_profiling

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


train=pd.read_csv('credit_train.csv')
test=pd.read_csv('credit_test.csv')


# # Combining two data set row wise

# In[23]:


#concatenate test and train
train['source']='train'# craeting new column and assign a value ('train') to help ourself in future to seprate them before modling
test['source']='test'
df = pd.concat([train,test],ignore_index=True, sort=True)
train.shape , test.shape,df.shape


# In[24]:


df.shape


# In[25]:


df.shape[0]


# In[26]:


df.shape[1]


# In[27]:


df.head()


# In[28]:


df.tail()


# In[24]:


#drop duplicate obs from train data set
train=train.drop_duplicates()


# In[25]:


#combining train and test data set to make a df_nodup data set
train['source']='train'
test['source']='test'
df_nodup = pd.concat([train,test],ignore_index=True, sort=True)
print(df.shape,df_nodup.shape,'\n Number of duplicate data : ',df.shape[0]-df_nodup.shape[0])


# In[26]:


#replace df with df_nodup
df=df_nodup


# # detect NaN and None with df.isnull() or df.isna()

# In[13]:


null=pd.isnull(train)
null.head()


# # Count the total number of missing values

# In[14]:


pd.isnull(train).sum().sum()


# # fill the missing value by zero

# In[15]:


train.fillna(0).head()


# # fill the list of missing vlaue with given value

# In[20]:


missing_value=("NA","",None,np.NaN)
missing=train.isin(missing_value)#detecting missing value
train.mask(missing,"missing").head()


# In[ ]:


from pandas_profiling import ProfileReport


# In[27]:


pandas_profiling.ProfileReport(df)


# # summary of DataFrame

# In[29]:


df.info()


# In[ ]:


Note:Python pandas.apply() is a member function in Dataframe class to apply a function along the axis of the Dataframe. 


# # Missing vlaue

# In[30]:


df.apply(lambda x: sum(x.isnull()))


# In[31]:


#calculatin no. of missing values for each column and it's percentage
def percentage_of_miss():
  df1=df[df.columns[df.isnull().sum()>=1]] # I did slicing by condition( I get s subset of dataframe that contains columns that have atleast one missing values) )
  total_miss = df1.isnull().sum().sort_values(ascending=False)
  percent_miss = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False) #df1.isnull().sum() returns only number of missing values,df1.isnull().count() returns whole number of observations (True=1 for null and False=0 for not missing ) 
  missing_data = pd.concat([total_miss, percent_miss], axis=1, keys=['Number of Missing', 'Percentage'])
  return(missing_data)


# In[32]:


percentage_of_miss()


# # to display column header

# In[35]:


pd.options.display.max_columns=100


# In[36]:


df.head()


# In[70]:


df.isna().mean().round(4)*100


# In[71]:


# alternative way
1-df.count()/len(df)


# In[46]:


df.tail()


# # how to handle outlier

# In[47]:


df.describe()


# In[49]:


pd.options.display.float_format = "{:.2f}".format


# In[50]:


df.describe()


# In[57]:


df.max()


# In[62]:


df[df['Annual Income']==df['Annual Income'].max()]


# In[63]:


df['Annual Income']


# In[68]:


df.set_index('Annual Income').select _dtypes('float')


# In[16]:


a=('hamid','shailendra','Adel','shailendra',455,None,True,45.5)
print(a)
print(type(a))
print(a[2])
print(len(a))


# # in tuple we can not change element but we can reassign

# In[18]:


a=['hamid','shailendra','Ram','shailendra',455,None,True,45.5]
print(a)


# In[22]:


b=['hamid','shailendra','Adel','shailendra',455,None,True,45.5]
print(b)
print(type(b))
print(b[2])
print(len(b))


# In[27]:


b[1]='hari'#list is changable(you can update/change the item(element)in the list)
print(b)


# # contructor 

# In[33]:


c=list(a)
print(a)
print(type(c))
d=tuple(a)
print(d)
print(type(d))


# In[34]:


s={'hamid','shailendra','Adel','shailendra',455,None,True,45.5}
print(s)
print(type(s))
print(len(s))


# # dictionary

# In[46]:


x={'Name':'hamid','Name':'ram','Name':'hari','age':40,'height':'186 cm', "male":True}
print(x)
print(type(x))
print(len(x))


# In[41]:


print(x['height'])


# In[45]:


x['age']=55
print(x)


# In[51]:


t="""My name is shailendra,
my son name is snehal.
i like to study DS with Hamid.
"""
print(t)
print(type(t))
print(len(t))
print(t[6])
print(t[0])


# In[52]:


t2="""my name is shailendra and I"""
len(t2)


# In[53]:


a="  shailndra  Berma"
a.strip()


# In[54]:


"shailendra berma".capitalize()


# In[56]:


a="  shailndra  Berma"


# In[57]:


a.replace("b","v")


# In[58]:


print(a.find("ma"))
print(a.index("ma"))


# In[59]:


print(a.find(" "))

