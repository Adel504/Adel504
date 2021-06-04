#!/usr/bin/env python
# coding: utf-8
Instructur Ms Gitimoni 

Prepared By Adel Hejazi 

May 02-2021 

# In[111]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, KDD, Knowledge Discovery Data')
os.getcwd()

# To change the current working directory in Python, use the chdir() method.
#The method accepts one argument, the path to the directory to which you want to change. 
#The path argument can be absolute or relative.


# In[112]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[113]:


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


# In[114]:


df=pd.read_csv("Car Evaluation.csv")     #By default keep_default_na=True
#If we don't use na_values='NA' here you won't get missing value for this data  ### ### the source of the data is 
# UCI Machine learning Repository 


# In[115]:


df.head(20) # Checking the  first 20 head of the dataset


# In[116]:


df.tail(20) # Checking the last 20 tail of the dataset


# In[117]:


df.shape # this give us the numer of rows and coulumns in yhe dataset 


# In[118]:


df.columns.values #  what are the columns values 


# In[119]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] # rename  the coulmns values with meaningful names 
#in the dataset 


# In[ ]:





# In[120]:


df.columns = col_names

col_names    # we dsiplay the columns name after we change the coulmns names 


# In[121]:


df.info() # inform about the data , data type 


# In[122]:


df['class'].value_counts() # here to see the count of class and what are the variables in it 


# In[123]:


df.size  # number of rows*number of columns 


# In[124]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:
    
    print(df[col].value_counts())    # we print the col value counts 


# In[125]:


df.isna().mean() # we are chcking the missing values in the dataframe df


# In[126]:


def Class_label(x):
    
    if x == 'unacc':
        return 0
    if x =='acc':
        return 1
    if x =='good':
        return 2
    if x =='vgood':
        return 3
    

# we change the catagoricilac variables in class to numeric from 0-3 to use it for plt.plot(fpr, tpr) and the  label Class  vs Rest
# In[127]:


df['class']=df['class'].apply(Class_label) # we change the class labels to numbers from 0 t 3 and apply it in the df


# In[128]:


X = df.drop(['class'], axis=1)

y = df['class']  # drop class and assign it to y to predict it 


# In[129]:


df['class'].value_counts() # the counts of class after change it to numbers 



# In[130]:


# split X and y into training and testing sets, after importing train_test_split from Sklearn 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[131]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[132]:


X_train.head() # we check the head of X_train 


# In[133]:


pip install --upgrade category_encoders 


# In[ ]:





# In[ ]:





# In[134]:



import category_encoders as ce  ## this pacake avaialbe in pythong to convert object datatype to numerical 


# In[135]:


# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[136]:


X_train.head() # here we see the head of X_train 


# In[137]:


y_test.head()  


# In[138]:


y_test.head()  # here we see the head of y_test 


# In[139]:


df.describe().transpose()#finding unique values, mode and fequency of mode
#transpose()=T


# In[140]:


df.keys()

keys() function returns the 'info axis' for the pandas object. If the pandas object is series then it returns index. If the pandas object is dataframe then it returns columns. If the pandas object is panel then it returns major_axis.
# In[141]:


#import Logistic Regression 
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500, tol=0.001)
model.fit(X_train,y_train)

Logistic Regression in Python With scikit-learn: Example 1. Import packages, functions, and classes. Get data to work with and, if appropriate, transform it. Create a classification model and train (or fit) it with your existing data. Evaluate your model to see if its performance is satisfactory.
# In[142]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train) #use only training set 
                                           #to make any adjustments to the model
                                           #during training

The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. In case of multivariate data, this is done feature-wise (in other words independently for each column of the data
# In[143]:


#Let's create a model again using the default settings
model = LogisticRegression()


# In[144]:


model.fit(X_train_scaled, y_train)  ## fit the model 

 fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a . predict() method call. To elaborate: Fitting your model to (i.e. using the . fit() method on) the training data is essentially the training part of the modeling process
# In[145]:


model.intercept_ #b0

b0 is the intercept of the regression line; that is the predicted value when x = 0 . b1 is the slope of the regression line.
# In[146]:


model.coef_ #coefficients of the features, b1, b2, ...


# In[147]:


#To be able to test we need to scale the test data too (X part only) 
#using the same scaler that was used to scale the training data
X_test_scaled = sc.transform(X_test)


# In[148]:


#Predict_proba gives the probabilities P(y=Ci|x)
probabilities_test = model.predict_proba(X_test_scaled)
probabilities_test[:5,1] #second column belongs to class 1, ie, p = P(y=1|x)


# In[149]:


#Whereas predict method gives the class prediction as either 0 or 1
y_predict = model.predict(X_test_scaled)
y_predict[:5]


# In[150]:


#Performance measures for classification
#Accuracy = total no. of correct prediction/total no. of datapoints

model.score(X_test_scaled,y_test)


# In[151]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,y_predict)

In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. ... Parameters y_true1d array-like, or label indicator array / sparse matrix. Ground truth (correct) labels.
# In[152]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)

print('Confusion matrix\n\n', cm)

A confusion matrix is a tabular summary of the number of correct and incorrect predictions made by a classifier. It can be used to evaluate the performance of a classification model through the calculation of performance metrics like accuracy, precision, recall, and F1-score.
# In[153]:


#k-fold cross-validation score 
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(max_iter=1000,tol=0.001),
                X_train_scaled, y_train,cv=4).mean().round(4)*100

Cross-validation is a statistical method used to estimate the skill of machine learning models. ... That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
# In[154]:


from sklearn.metrics import classification_report   ## here we import classification_report from sklearn 
print(classification_report(y_test,y_predict))

When F1 score is 1 it’s best and on 0 it’s worst. F1 = 2 * (precision * recall) / (precision + recall) Precision and Recall should always be high.
# In[ ]:




A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below.
# In[155]:



# multi-class classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

ROC or Receiver Operating Characteristic curve is used to evaluate logistic regression classification models.
AUC or area under the curve 
# In[156]:



# fit model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)

One-vs-the-rest (OvR) multiclass strategy. Also known as one-vs-all, this strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes.
# In[157]:



# roc curve for classes
fpr = {}
tpr = {}
thresh ={}


n_class = 4

for i in range (n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test,  pred_prob[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0  vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='yellow',label = 'Class 3 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);  
plt.title('Roc with auc score: {}'.format(roc_auc_score(y_test,clf.predict_proba(X_test_scaled),multi_class='ovr')))
plt.show()


# from PIL import Image, ImageDraw

# In[ ]:


Conclusion 


# In[158]:


# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[160]:


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

rfc_100.fit(X_train, y_train)



# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)



# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


# In[162]:


# instantiate the classifier with n_estimators = 1000

rfc_100 = RandomForestClassifier(n_estimators=1000, random_state=0)



# fit the model to the training set

rfc_100.fit(X_train, y_train)



# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)



# Check accuracy score 

print('Model accuracy score with 1000 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


# In[ ]:





# In[163]:


# create the classifier with n_estimators = 100

clf = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

clf.fit(X_train, y_train)


# In[164]:


# view the feature scores

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores


# In[165]:


# Creating a seaborn bar plot

sns.barplot(x=feature_scores, y=feature_scores.index)



# Add labels to the graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')



# Add title to the graph

plt.title("Visualizing Important Features")



# Visualize the graph

plt.show()


# In[173]:


# declare feature vector and target variable

X = df.drop(['class', 'doors'], axis=1)

y = df['class']


# In[174]:


# split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[175]:


# encode categorical variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[176]:


# instantiate the classifier with n_estimators = 100

clf = RandomForestClassifier(random_state=0)



# fit the model to the training set

clf.fit(X_train, y_train)


# Predict on the test set results

y_pred = clf.predict(X_test)



# Check accuracy score 

print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[177]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[178]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:


Conclusion 

I build a Random Forest Classifier to predict the safety of the car. I build two models, one with 100 decision-trees and another one with 1000 decision-trees.The model accuracy score with 100 decision-trees is : 0.9615 but the same with 1000 decision-trees is  0.9672 . So, as expected accuracy increases with number of decision-trees in the model.WE did score for important features, and visualize it using seaborn 
we drop the least value, "doors", but the the score decrease, i just show it 
# # Thank you 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ROC-AUC score is one of the major metrics to assess the performance of a classification model. 
# Logistic Regression Model as it is robust against probability threshold values and truly depicts if the model is good or not for the data at hand. The closer the score to 1, the better. If the score is near 0.5, it means that Logistic Regression is not a good fit for the data. Either we need to get more discriminative features to help identify the target class or look for other model options (may be a complex non-linear model)
