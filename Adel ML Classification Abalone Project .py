#!/usr/bin/env python
# coding: utf-8

# In[38]:





# In[ ]:





# In[21]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, Machine Learning')
os.getcwd()

# To change the current working directory in Python, use the chdir() method.
#The method accepts one argument, the path to the directory to which you want to change. 
#The path argument can be absolute or relative.


# In[22]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[23]:


# Importing the numpy and pandas package
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# multi-class classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings(action='ignore')


# In[24]:


df=pd.read_csv("abalone.csv")     #By default keep_default_na=True
#If we don't use na_values='NA' here you won't get missing value for this data  ### ### the source of the data is 
# UCI Machine learning Repository 


# In[25]:


#here we are chining about missing values 
df.isna().sum()


# In[26]:


# here we are dropping missing values and copying the data to work on the copy not in the original data 
data=df.dropna(how='all')


# In[27]:


#here we are counting the missing values
data.isna().sum()


# In[28]:


# info about the data and data type 

data.info()


# In[29]:


# show the attribute sex in the data
data.Sex


# In[30]:


# show the correlation 

data.corr()


# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


# In[32]:


# dropping missing values 

data.dropna()


# In[33]:


# we drop sex as it will be the target 

X = data.drop(['Sex'], axis=1)

y = data['Sex']


# In[34]:


# split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[35]:


# show first 5 head of X_train 
X_train.head()


# In[36]:


# show X_test 5 head 

X_test.head()


# In[37]:


# show df 

df


# In[38]:


# data shape rows and colums 

df.shape


# In[39]:


df.describe()


# In[40]:


df.isna().sum()


# In[41]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[42]:


import category_encoders as ce


# In[44]:


# encoding all variables and split to X_train , and X_test BEFORE MODELING 

encoder = ce.OrdinalEncoder(cols=['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weightShell weight','Shell weight','Rings'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[45]:


X_train.head()


# In[46]:


X_test.head()


# In[47]:


# convert text labels to integer labels
sex_label = LabelEncoder()
data['Sex'] = sex_label.fit_transform(data['Sex'])
data.head()


# In[49]:


# show the sex variables unique values 

data['Sex'].value_counts()


# In[51]:


#scale the features using training set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = pd.DataFrame(sc.fit_transform(X_train),columns=X.columns)
X_test_scaled = pd.DataFrame(sc.transform(X_test),columns=X.columns)


# In[52]:


#import the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500,tol=0.001)


# In[53]:


#import Logistic Regression 
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500, tol=0.001)
model.fit(X_train_scaled, y_train)


# In[54]:


model.intercept_


# In[55]:


model.coef_


# In[56]:


#Validating the model
#Performance measures for classification
#Accuracy = total no. of correct prediction/total no. of instances
model.score(X_test_scaled,y_test)


# In[ ]:


#k-fold cross-validation score
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(max_iter=1000,tol=0.001),
                X_train_scaled, y_train,cv=5).mean().round(4)*100


# In[57]:


#import the knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[58]:


#see the cross_validated score for cv=3
from sklearn.model_selection import cross_val_score
cross_val_score(knn,X,y,cv=3).mean()


# In[59]:


#for no.of neighbors from 1 - 10, graph the k-fold scores
scores = []
for i in range(1,11,1):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    scores.append(cross_val_score(knn,X,y,cv=3).mean())


# In[60]:


import matplotlib.pyplot as plt
plt.plot(range(1,11,1),scores)
plt.xlabel('no. of neighbors')
plt.ylabel('k-fold test scores')
plt.show()


# # 7-NN is the best

# In[48]:


# we are applying SVM MODEL support vector machine 


# In[61]:


#importing from Sklearn 
from sklearn.svm import SVC

from sklearn import metrics, svm


# In[63]:


# importing Grid search from sklearn to get teh best model performance parameters 
from sklearn.model_selection import GridSearchCV


# In[64]:


params_dictionary = {
                        'C' : [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2,3],
                        'gamma' : [0.1,1,10]
                    }

model = GridSearchCV(SVC(random_state=0),param_grid=params_dictionary,cv=10)


# In[65]:


params_dictionary = {
                        'C' : [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2,3],
                        'gamma' : [0.1,1,10]
                    }

model = GridSearchCV(SVC(random_state=0),param_grid=params_dictionary,cv=4)


# In[ ]:


# fir the model 
model.fit(X,y) # 


# In[ ]:


# best parameter 
model.best_params_


# In[ ]:


# best model 
model.best_score_


# In[ ]:


svm =model.best_estimator_


# In[ ]:


# fit svm 
svm.fit(X_train,y_train)


# In[ ]:


# SVM score 
svm.score(X_test,y_test)


# # ADABOOST

# In[ ]:


# applying ADAboost after importing from sklearn 
from sklearn.ensemble import AdaBoostClassifier


#Graph k-fold score vs no. of estimators in Adaboost which uses DT as base estimators
scores = []
for i in range(10,110,10):
    scores.append(cross_val_score(AdaBoostClassifier(n_estimators=i,random_state=0),
                                  X,y,cv=10).mean())
plt.plot(range(10,110,10),scores)
plt.xlabel('No. of DTs in Adaboost')
plt.ylabel('K-fold scores')
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#including other params like max_depth, we will apply gridsearch to fine the best settings 
params = {
            'n_estimators': [70,80,90,100],
            'base_estimator': [DecisionTreeClassifier(max_depth=9,random_state=0),
                               DecisionTreeClassifier(max_depth=10,random_state=0),
                               DecisionTreeClassifier(max_depth=11,random_state=0)]
        }
model = GridSearchCV(AdaBoostClassifier(random_state=0), params,cv=10)
model.fit(X,y)


# In[ ]:


# best nodel 
model.best_params_


# In[106]:


# best score 
model.best_score_


# In[ ]:


# importing the necessary packages 
to apply random forestclassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[109]:


# define score 
scores = []
for i in range(10,101,10):
    scores.append(cross_val_score(RandomForestClassifier(n_estimators=i,random_state=9),
                                  X,y,cv=4).mean())


# In[ ]:


# show the plot diagram for random forest 
plt.plot(range(10,101,10),scores)
plt.xlabel('No. of DTs in RandomForest')
plt.ylabel('K-fold scores')
plt.show()


# In[ ]:


params = {
            'n_estimators': [100,110,120,130],
            'max_depth': [13,14,15]
        }
model = GridSearchCV(RandomForestClassifier(), params,cv=10)
model.fit(X,y)


# In[113]:


model.best_params_


# In[114]:


model.best_score_


# In[115]:


best_model = model.best_estimator_


# In[116]:


# importing sklearn.model_selection from sklearn 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=5)


# In[117]:


best_model.fit(X_train,y_train)


# In[118]:


best_model.score(X_test,y_test)


# In[119]:


best_model = model.best_estimator_


# In[120]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[121]:


best_model.fit(X_train,y_train)


# In[122]:


y_pred = best_model.predict(X_test)


# In[123]:


# printing the classification report 
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#https://www.kaggle.com/abedkurdi/abalone-randomforestclassifier

