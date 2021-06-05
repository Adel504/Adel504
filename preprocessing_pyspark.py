#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
 
os.environ["SPARK_HOME"] = "/usr/hdp/current/spark2-client"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
# In below two lines, use /usr/bin/python2.7 if you want to use Python 2
os.environ["PYSPARK_PYTHON"] = "/usr/local/anaconda/bin/python" 
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/anaconda/bin/python"
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.10.4-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")


# In[11]:


from pyspark.sql import SparkSession
spark = SparkSession     .builder     .appName("SparkSQL and SparkML")     .getOrCreate()


# In[3]:


df = spark.read.csv('Data.csv',header=True,inferSchema=True)


# In[4]:


df.show()


# In[5]:


df.filter(df['Salary'].isNull()).show()


# In[6]:


for col in df.columns:
    print("no. of cells in column", col, "with null values:", df.filter(df[col].isNull()).count())


# In[7]:


df.dropna().show()


# In[8]:


df.show()


# In[9]:


df.fillna({'Age':38,'Salary':67000}).show()


# In[10]:


from pyspark.sql.functions import avg
def mean_of_pyspark_columns(df, numeric_cols, verbose=False):
    col_with_mean={}
    for col in numeric_cols:
        mean_value = df.select(avg(df[col]))
        avg_col = mean_value.columns[0]
        res = mean_value.rdd.map(lambda row : row[avg_col]).collect()
        
        if (verbose==True): print(mean_value.columns[0], "\t", res[0])
        col_with_mean[col]=res[0]    
    return col_with_mean


# In[11]:


mean_of_pyspark_columns(df,['Age','Salary'])


# In[12]:


df = df.fillna(mean_of_pyspark_columns(df,['Age','Salary']))


# In[13]:


df.show()


# In[14]:


def mode_of_pyspark_columns(df, cat_col_list, verbose=False):
    col_with_mode={}
    for col in cat_col_list:
        #Filter null
        df = df.filter(df[col].isNull()==False)
        #Find unique_values_with_count
        unique_classes = df.select(col).distinct().rdd.map(lambda x: x[0]).collect()
        unique_values_with_count=[]
        for uc in unique_classes:
             unique_values_with_count.append([uc, df.filter(df[col]==uc).count()])
        #sort unique values w.r.t their count values
        sorted_unique_values_with_count= sorted(unique_values_with_count, key = lambda x: x[1], reverse =True)
        
        if (verbose==True): print(col, sorted_unique_values_with_count, " and mode is ", sorted_unique_values_with_count[0][0])
        col_with_mode[col] = sorted_unique_values_with_count[0][0]
    return col_with_mode


# In[15]:


mode_of_pyspark_columns(df,["Country","Purchased"],True)


# In[16]:


df.fillna(mode_of_pyspark_columns(df,["Country","Purchased"])).show()


# In[17]:


#Label encoder
from pyspark.ml.feature import StringIndexer
indexed = df
for col in ["Purchased","Country"]:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col+"_encoded")
    indexed = stringIndexer.fit(indexed).transform(indexed)
indexed.show()


# In[18]:


#One hot encoder
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol="Country_encoded",outputCol="Country_vec",dropLast=True)
encoded = encoder.transform(indexed)
encoded.show()


# In[19]:


from pyspark.ml.feature import VectorAssembler, StandardScaler
assembler = VectorAssembler(inputCols=["Age","Salary","Country_vec"], 
                            outputCol="features")
feature_vec=assembler.transform(encoded)
feature_vec.select("features").take(3)


# In[20]:


feature_vec.show(3)


# In[21]:


# Split the data into train and test sets
train_data, test_data = feature_vec.randomSplit([.8,.2],seed=1234)


# In[22]:


train_data.show(1)


# In[23]:


from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scalerModel = scaler.fit(train_data)
scaledData = scalerModel.transform(train_data)
scaledData_test = scalerModel.transform(test_data)
scaledData.select("scaledFeatures").take(3)


# In[7]:


spark.stop()


# In[ ]:




