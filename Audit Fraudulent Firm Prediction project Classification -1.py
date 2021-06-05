#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Audit Fraudulent Firm Prediction project Classification 

Instructur Ms Gitimoni 

Prepared By Adel Hejazi 

May 25-2021 
# In[ ]:




Fraudulent Firm Prediction
Given data about Audits of firms, let's try to predict whether a given firm will be fraudulent or not.

The goal of the dataset is to help the Auditors by building a classification model that can predict the fraudulent firm on the basis the present and historical risk factors.
 

We will use a variety of classification models to make our predictions.
# In[1]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, Machine Learning')
os.getcwd()

# To change the current working directory in Python, use the chdir() method.
#The method accepts one argument, the path to the directory to which you want to change. 
#The path argument can be absolute or relative.


# In[2]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[3]:


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


# In[4]:


df=pd.read_csv("audit_data.csv")     #By default keep_default_na=True
#If we don't use na_values='NA' here you won't get missing value for this data  ### ### the source of the data is 
# UCI Machine learning Repository 


# In[5]:


df.columns.values


# In[6]:


df.info()


# In[7]:


df.to_csv('test1.csv')


# In[ ]:





# In[8]:


df


# In[9]:


df.shape


# In[10]:


#There are some values with "?"  in categorical variables

df1 = df.replace('?',np.NaN)
df1.isnull().any()


# In[11]:


def preprocess_inputs(df):
    df = df.copy()


# In[12]:


# Fill missing value with the mean 
df['Money_Value'] = df['Money_Value'].fillna(df['Money_Value'].mean())


# In[13]:


# One-hot encode the LOCATION_ID column
location_dummies = pd.get_dummies(df['LOCATION_ID'], prefix='location')
df = pd.concat([df, location_dummies], axis=1)
df = df.drop('LOCATION_ID', axis=1)
    
   


# In[14]:


# Split df into X and y   ## here risk is the target 
   
y = df['Risk']
X = df.drop('Risk', axis=1)
 
   


# In[15]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)


# In[16]:


# Scale X
scaler = StandardScaler()
scaler.fit(X_train)
X_train =pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
 


# In[17]:


X_train


# In[18]:


y_train


# In[19]:


models = {
    "                   Logistic Regression": LogisticRegression(),
    "                   K-Nearest Neighbors": KNeighborsClassifier(),
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier(),
    "                         Random Forest": RandomForestClassifier(),
    "                     Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained.")


# In[20]:


for name, model in models.items():
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))


# In[21]:


#import Logistic Regression 
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500, tol=0.001)
model.fit(X_train,y_train)


# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train) #use only training set 
                                           #to make any adjustments to the model
                                           #during training


# In[23]:


#Let's create a model again using the default settings
model = LogisticRegression()


# In[24]:


model.fit(X_train_scaled, y_train)  ## fit the model 


# In[25]:


model.fit(X_train_scaled, y_train)  ## fit the model 


# In[26]:


model.fit(X_train_scaled, y_train)  ## fit the model 


# In[27]:


model.coef_ #coefficients of the features, b1, b2, ...


# In[28]:


model.intercept_ #b0


# In[29]:


#To be able to test we need to scale the test data too (X part only) 
#using the same scaler that was used to scale the training data
X_test_scaled = sc.transform(X_test)


# In[30]:


#Predict_proba gives the probabilities P(y=Ci|x)
probabilities_test = model.predict_proba(X_test_scaled)
probabilities_test[:5,1] #second column belongs to class 1, ie, p = P(y=1|x)


# In[31]:


#Whereas predict method gives the class prediction as either 0 or 1
y_predict = model.predict(X_test_scaled)
y_predict[:5]


# In[32]:


#Performance measures for classification
#Accuracy = total no. of correct prediction/total no. of datapoints

model.score(X_test_scaled,y_test)


# In[33]:


#k-fold cross-validation score 
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(max_iter=1000,tol=0.001),
                X_train_scaled, y_train,cv=4).mean().round(4)*100


# In[74]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[35]:


pipelineLR = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, penalty='l2', solver='lbfgs'))
#
# Create the parameter grid
#
param_grid_lr = [{
    'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
}]
#
# Create an instance of GridSearch Cross-validation estimator
#
gsLR = GridSearchCV(estimator=pipelineLR,
                     param_grid = param_grid_lr,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)


# In[36]:


#
# Train the LogisticRegression Classifier
#
gsLR = gsLR.fit(X_train, y_train)
#
# Print the training score of the best model
#
print(gsLR.best_score_)
#
# Print the model parameters of the best model
#
print(gsLR.best_params_)
#
# Print the test score of the best model
#
clfLR = gsLR.best_estimator_
print('Test accuracy: %.3f' % clfLR.score(X_test, y_test))


# In[37]:


gsLR.fit(X,y)


# In[38]:


gsLR.best_params_


# In[39]:


gsLR.best_score_


# In[40]:


y_pred = gsLR.predict(X_test)


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[42]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[43]:


#import the knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[44]:


#see the cross_validated score for cv=3
from sklearn.model_selection import cross_val_score
cross_val_score(knn,X,y,cv=3).mean()


# In[45]:


# K-Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsClassifier 
RegModel = KNeighborsClassifier(n_neighbors=4)

# Printing all the parameters of KNN
print(RegModel)

# Creating the model on Training Data
KNN=RegModel.fit(X_train,y_train)
prediction=KNN.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, KNN.predict(X_train)))

# Plotting the feature importance for Top 10 most important columns
# The variable importance chart is not available for KNN


# In[46]:


#for no.of neighbors from 1 - 10, graph the k-fold scores
scores = []
for i in range(1,11,1):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    scores.append(cross_val_score(knn,X,y,cv=3).mean())


# In[62]:


import matplotlib.pyplot as plt
plt.plot(range(1,11,1),scores)
plt.xlabel('no. of neighbors')
plt.ylabel('k-fold test scores')
plt.show()


# In[63]:


from sklearn.pipeline import make_pipeline


# In[64]:


# Using Grid Search KNN TUNING:


knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

#Define the grid of hyperparameters
params_grid = {'n_neighbors': [3,5,10],
              'weights': ['uniform']
              }

#Initiate Grid search
grid_model = GridSearchCV(estimator = knn_model, param_grid = params_grid , cv = 3)
                       
#Fitting the grid search
grid_model.fit(X_train, y_train)


# In[65]:


grid_model.fit(X,y)


# In[66]:


grid_model.best_params_


# In[67]:


grid_model.best_score_


# In[68]:


y_pred = grid_model.predict(X_test)


# In[69]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[70]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[77]:


#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC # support victor classifier 


# In[78]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[79]:


from sklearn.model_selection import GridSearchCV


# In[80]:


params_dictionary = {
                        'C' : [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2,3],
                        'gamma' : [0.1,1,10]
                    }

model = GridSearchCV(SVC(random_state=0),param_grid=params_dictionary,cv=4)


# In[81]:


model.fit(X,y)


# In[82]:


model.best_params_


# In[83]:


model.best_score_


# In[84]:


svm =model.best_estimator_


# In[85]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[86]:


svm.fit(X_train,y_train)


# In[87]:


svm.score(X_test,y_test)


# In[60]:


scores = []
for i in range(1,11,1):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    scores.append(cross_val_score(knn, X = X_train, y = y_train, cv = 10).mean())
    
plt.plot(range(1,11,1),scores)
plt.xlabel('no. of neighbors')
plt.ylabel('k-fold test scores')
plt.show()


# In[88]:


from sklearn.model_selection import GridSearchCV


# In[89]:


params_dictionary = {
                        'C' : [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2,3],
                        'gamma' : [0.1,1,10]
                    }

model = GridSearchCV(SVC(random_state=0),param_grid=params_dictionary,cv=4)


# In[90]:


model.fit(X,y)


# In[91]:


model.best_params_


# In[92]:


model.best_score_


# In[93]:


y_pred = model.predict(X_test)


# In[94]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[95]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[96]:


# Load libraries
from sklearn.ensemble import AdaBoostClassifier


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[97]:


plt.plot(range(10,101,10),scores)
plt.xlabel('No. of DTs in Adaboost')
plt.ylabel('K-fold scores')
plt.show()


# In[98]:


# ADABOOST TUNING
params = {
            'n_estimators': [0.1,1,5,10],
            'base_estimator': [DecisionTreeClassifier(max_depth=13,random_state=0),
                               DecisionTreeClassifier(max_depth=14,random_state=0),
                               DecisionTreeClassifier(max_depth=16,random_state=0)]
        }
grid_model5 = GridSearchCV(AdaBoostClassifier(random_state=0), params,cv=15)
grid_model5.fit(X,y)


# In[99]:


grid_model5.fit(X,y)


# In[100]:


grid_model5.best_params_


# In[101]:


grid_model5.best_score_


# In[102]:


y_pred = grid_model5.predict(X_test)


# In[103]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[104]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[105]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[106]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[107]:


scores = []
for i in range(10,101,10):
    scores.append(cross_val_score(RandomForestClassifier(n_estimators=i,random_state=9), X,y,cv=10).mean())


# In[108]:


plt.plot(range(10,101,10),scores)
plt.xlabel('No. of DTs in RandomForest')
plt.ylabel('K-fold scores')
plt.show()


# In[109]:


pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=1))
#
# Create the parameter grid
#
param_grid_rfc = [{
    'randomforestclassifier__max_depth':[2, 3, 4],
    'randomforestclassifier__max_features':[2, 3, 4, 5, 6]
}]
#
# Create an instance of GridSearch Cross-validation estimator
#
gsRFC = GridSearchCV(estimator=pipelineRFC,
                     param_grid = param_grid_rfc,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)


# In[110]:



# Train the RandomForestClassifier
#
gsRFC = gsRFC.fit(X_train, y_train)
#
# Print the training score of the best model
#
print(gsRFC.best_score_)
#
# Print the model parameters of the best model
#
print(gsRFC.best_params_)
#
# Print the test score of the best model
#
clfRFC = gsRFC.best_estimator_
print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))


# In[111]:


gsRFC.fit(X,y)


# In[112]:


gsRFC.best_params_


# In[113]:


gsRFC.best_score_


# In[114]:


y_pred = gsRFC.predict(X_test)


# In[115]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[116]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

Conclusion: we can see the score  were varied from model to model and by using Grid search  improves  performance. we can see all score are between 92% -99.3%, which give company more than one choice to choose from as most of  all models used here got good score , however we recommend  Random forest classificatin because it gives good result even without hyper-Prameter Tuning
its also because of the simplicity and diversity 
# In[ ]:


**************Thank you *******Any Questions ***********************


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




