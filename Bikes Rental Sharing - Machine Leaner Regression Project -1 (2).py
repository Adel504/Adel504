#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Bikes Rental sharing- machine Learning Project 

Instructur Ms Gitimoni 

Prepared By Adel Hejazi 

May 25-2021 This dataset contains the hourly and daily count of rental bikes between the years 2011 and 2012 in the Capital bike share system with the corresponding weather and seasonal information.The target is to count Total Rental Bikes including Casual,Registered and correlations between the Bikes rental and other variables like Year, month, hour,wather, events
# In[313]:


import os      # Change Current Working Directory in Python # os stand for operating system in python 
os.chdir(r'D:\Data Mining, Machine Learning')
os.getcwd()

# To change the current working directory in Python, use the chdir() method.
#The method accepts one argument, the path to the directory to which you want to change. 
#The path argument can be absolute or relative.


# In[139]:


import warnings      # In order to disable all warnings in the current script/notebook just use filterwarnings
warnings.filterwarnings('ignore')  


# In[186]:


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
from numpy import absolute


# In[141]:


df=pd.read_csv("BikeSharingRental.csv")     #By default keep_default_na=True
#If we don't use na_values='NA' here you won't get missing value for this data  ### ### the source of the data is 
# UCI Machine learning Repository 


# In[142]:


df.head()


# In[143]:


df.tail()


# In[144]:


df.describe()


# In[145]:


df.shape


# In[146]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creating Bar chart as the Target variable is Continuous
df['cnt'].hist()


# In[147]:


# We already have relevant info on date with yr, month and hour
# and we want only the total count
# also instant is the irrelevant for prediction
pre_dropped = ["dteday", "casual",  "instant"]
df = df.drop(pre_dropped, axis=1)
df.isnull().sum() # no missing data


# In[148]:


df.isnull().sum()


# In[149]:


df.shape


# In[150]:


# Unique values:
df.apply(lambda x: len(x.unique()))


# In[151]:


# data columns and data type
df.info()


# In[152]:


# let's check if numerical features are correlated with one another
sns.heatmap(df[["temp", "atemp", "windspeed", "hum", "cnt"]].corr(), annot=True)


# In[153]:


# Calculating correlation matrix
ContinuousCols=['cnt','temp','atemp','hum','windspeed','registered']

# Creating the correlation matrix
CorrelationData=df[ContinuousCols].corr()
CorrelationData


# In[154]:


# Filtering only those columns where absolute correlation > 0.5 with Target Variable
# reduce the 0.5 threshold if no variable is selected
CorrelationData['cnt'][abs(CorrelationData['cnt']) > 0.5 ]


# In[155]:


#Based on the above tests, selecting the final columns for machine learning

SelectedColumns=['registered','season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']

# Selecting final columns
DataForML=df[SelectedColumns]
DataForML.head()


# In[156]:


#In this data there is no Ordinal categorical variable which is in string format.

#Converting the binary nominal variable to numeric using 1/0 mapping
#All the binary nominal variables are already in numeric format

#Converting the nominal variable to numeric using get_dummies()
# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['cnt']=df['cnt']

# Printing sample rows
DataForML_Numeric.head()


# In[157]:


#Machine Learning: Splitting the data into Training and Testing sample
#We dont use the full data for creating the model. Some data is randomly selected and kept aside for checking how good the model is. This is known as Testing Data and the remaining data is called Training data on which the model is built. Typically 70% of data is used as Training data and the rest 30% is used as Tesing data.

# Printing all the column names for our reference
DataForML_Numeric.columns


# In[158]:


# Separate Target Variable and Predictor Variables
TargetVariable='cnt'
Predictors=['registered', 'season', 'mnth', 'hr', 'holiday',
       'weekday', 'workingday', 'weathersit']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)


# In[159]:


# correlation Matrix using Person Correlation Matrix as we can see there is postive linear correaltion between registered and count for bikes
matrix_correlation = df.corr(method='pearson')
n_ticks = len(df.columns)
plt.figure( figsize=(9, 9) )
plt.xticks(range(n_ticks), df.columns, rotation='vertical')
plt.yticks(range(n_ticks), df.columns)
plt.colorbar(plt.imshow(matrix_correlation, interpolation='nearest', 
                            vmin=-1., vmax=1., 
                            cmap=plt.get_cmap('Blues')))
_ = plt.title("Pearson's Correlation Matrix")


# In[160]:


#Standardization/Normalization of data
#You can choose not to run this step if you want to compare the resultant accuracy of this transformation with the accuracy of raw data.

#However, if you are using KNN or Neural Networks, then this step becomes necessary.

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[161]:


# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[184]:


# importing all neceassy packages we need to use from Sklearn 
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

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import KFold
from numpy import mean


# In[164]:


#importing OLS statsmodel to check the p-values of the X variable (Ordinary least squares)
import statsmodels.api as sm
X2 = sm.add_constant(X) 
ols = sm.OLS(y,X2)
lr = ols.fit()
print(lr.summary())

In K-Folds Cross Validation we split our data into k different subsets (or folds). We use k-1 subsets to train our data and leave the last subset (or the last fold) as test data. We then average the model against each of the folds and then finalize our model. After that we test it against the test set.
# In[165]:


#k-fold cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(LinearRegression(),X,y,cv=5)


# In[166]:


from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error


# In[167]:


y_pred = RegModel.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
import math

print(r2_score(y_test,y_pred)) #R-squared (R 2) is an important statistical measure which in a regression model represents
# the proportion of the difference or variance  for a dependent variable which can be explained by an independent variable or variables. .


# In[168]:


print(mean_squared_error(y_test,y_pred)) #MSE (Mean squared Error.)


# In[169]:


print(math.sqrt(mean_squared_error(y_test,y_pred))) #RMSE # root mean square error 


# In[198]:



# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
RegModel = LinearRegression()

# Printing all the parameters of Linear regression
print(RegModel)

# Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)
prediction=LREG.predict(X_test)

# Taking the standardized values to original scale


from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))


# In[199]:


print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['cnt']-TestingDataResults['Predictedcnt']))/TestingDataResults['cnt'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[200]:


model = LinearRegression()


# In[201]:


#define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)


# In[202]:


#use k-fold CV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)


# In[203]:



#view mean absolute error
mean(absolute(scores))


# In[204]:


#k-fold cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(LinearRegression(),X,y,cv=5)


# In[206]:


# GRADIANT DESCENT

params = {
            'loss': ['squared_loss'],
            'penalty':['elasticnet'],
            'alpha': [0.1],
            'l1_ratio':[1],
            'learning_rate':['optimal'],
            'eta0':[0.001],
            'power_t':[0.01]
         }
grid_model = GridSearchCV(SGDRegressor(random_state=7), params, cv=4)
grid_model.fit(X_train,y_train)


# In[208]:


# GRADIANT DESCENT

lr_sgd = grid_model.best_estimator_

lr_sgd.fit(X_train,y_train)

pred = lr_sgd.predict(X_test)


# In[209]:


#  MODEL INTERCEPT
lr_sgd.intercept_


# In[211]:


SSE = np.sum((y_test-pred)**2)
SST = np.sum((y_test-np.mean(pred))**2)
SSR = SST-SSE
R2 =1-(SSE/SST)


# In[214]:


print('Best Parameters :')
print(grid_model.best_params_)
print('')

print('Best Score      : %.2f'%grid_model.best_score_)
print('Test R^2        : %.2f'%grid_model.best_estimator_.score(X_test, y_test))
print('Training R^2    : %.2f'%grid_model.best_estimator_.score(X_train, y_train))
print('')
print('Model Fit:')
print('R2        :%.2f'%r2_score(y_test,pred)) #R^2
print('RMSE      :%.2f'%math.sqrt(mean_squared_error(y_test, pred))) #Root-Mean Square Error (RMSE)


# In[241]:


import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from numpy import sqrt
import pandas as pd


# In[242]:


#shuffle the data
from sklearn.utils import shuffle
X,y = shuffle(X,y,random_state=0)


# In[243]:


# K-Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsRegressor
RegModel = KNeighborsRegressor(n_neighbors=4)

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


# In[218]:


print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['cnt']-TestingDataResults['Predictedcnt']))/TestingDataResults['cnt'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),4))


# In[250]:


#KNN TUNING:

#Initiating Random Forest regressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform')

#Define the grid of hyperparameters
params_grid = {'n_neighbors': [3,5,10],
              'weights': ['uniform']
              }

#Initiate Grid search
grid_model = GridSearchCV(estimator = knn_model, param_grid = params_grid , cv = 3)
                       
#Fitting the grid search
grid_model.fit(X_train, y_train)


# In[252]:


# GRADIANT DESCENT
knn_opt = grid_model.best_estimator_
knn_opt.fit(X_train,y_train)
pred = knn_opt.predict(X_test)


# In[254]:


SSE = np.sum((y_test-pred)**2)
SST = np.sum((y_test-np.mean(pred))**2)
SSR = SST-SSE
R2 =1-(SSE/SST)


# In[255]:


print('Best Parameters :')
print(grid_model.best_params_)
print('')

print('Best Score      : %.2f'%grid_model.best_score_)
print('Test R^2        : %.2f'%grid_model.best_estimator_.score(X_test, y_test))
print('Training R^2    : %.2f'%grid_model.best_estimator_.score(X_train, y_train))
print('')
print('Model Fit:')
print('R2        :%.2f'%r2_score(y_test,pred)) #R^2
print('RMSE      :%.2f'%math.sqrt(mean_squared_error(y_test, pred))) #Root-Mean Square Error (RMSE)


# In[263]:


from sklearn.svm import LinearSVR

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


# In[264]:


import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle


# In[265]:


# Fitting SVR to the dataset 
from sklearn.svm import SVR 
regressor = SVR(kernel = 'rbf') 
regressor.fit(X, y)


# In[266]:


lsvr = LinearSVR(verbose=0, dual=True)
print(lsvr)

LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
          intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
          random_state=None, tol=0.0001, verbose=0)


# In[267]:


#For cross-validation using train-test X, Y  split for 20% test  and 80%  train 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 random_state=1,test_size=0.2)


# In[268]:


lsvr.fit(X_train, y_train)

score = lsvr.score(X_train, y_train)
print(score)


# In[269]:


cv_score = cross_val_score(lsvr, X, y, cv = 10)
print("CV mean score: ", cv_score.mean())


# In[270]:


from sklearn.svm import LinearSVR, NuSVR, OneClassSVM


linear_svr = LinearSVR(max_iter=1000)
linear_svr.fit(X_train, y_train)


# In[271]:


y_preds = linear_svr.predict(X_test)

print(y_preds[:10])
print(y_test[:10])

print('Test R^2 Score : %.3f'%linear_svr.score(X_test, y_test)) ## Score method also evaluates accuracy for classification models.
print('Training R^2 Score : %.3f'%linear_svr.score(X_train, y_train))


# In[272]:


print("Feature Importances :", linear_svr.coef_)


# In[273]:


print("Model Intercept :", linear_svr.intercept_)


# In[274]:


lsvr = LinearSVR()
lsvr.fit(X_train, y_train)


# In[275]:


score = lsvr.score(X_train, y_train)
print("R-squared:", score)


# In[277]:


# using Grid search to tune the hyper parameter  SVR TUNING: support vector regession 

params_grid = {
                'kernel' : ['rbf'],   
                'C' : [1],
                'degree' : [3],
                'epsilon':[0.1],
                'gamma' : ['auto']
         }
grid_model = GridSearchCV(SVR(),params_grid)
grid_model.fit(X_train,y_train)


# In[279]:


# GRADIANT DESCENT

svr_sgd = grid_model.best_estimator_
svr_sgd.fit(X_train,y_train)
pred = svr_sgd.predict(X_test)


# In[281]:


# sum square error,sum square total, sum sqaure regression 
SSE = np.sum((y_test-pred)**2)
SST = np.sum((y_test-np.mean(pred))**2)
SSR = SST-SSE
R2 =1-(SSE/SST)


# In[283]:


print('Best Parameters :')
print(grid_model.best_params_)
print('')

print('Best Score      : %.2f'%grid_model.best_score_)
print('Test R^2        : %.2f'%grid_model.best_estimator_.score(X_test, y_test))
print('Training R^2    : %.2f'%grid_model.best_estimator_.score(X_train, y_train))
print('')
print('Model Fit:')
print('R2        :%.2f'%r2_score(y_test,pred)) #R^2
print('RMSE      :%.2f'%math.sqrt(mean_squared_error(y_test, pred))) #Root-Mean Square Error (RMSE)


# In[288]:


# Adaboost (Boosting of multiple Decision Trees)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Choosing Decision Tree with 1 level as the weak learner
DTR=DecisionTreeRegressor(max_depth=10)
RegModel = AdaBoostRegressor(n_estimators=100, base_estimator=DTR ,learning_rate=0.04)

# Printing all the parameters of Adaboost
print(RegModel)

# Creating the model on Training Data
AB=RegModel.fit(X_train,y_train)
prediction=AB.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, AB.predict(X_train)))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(AB.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# In[287]:


print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['cnt']-TestingDataResults['Predictedcnt']))/TestingDataResults['cnt'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[289]:


# ADABOOST TUNING
params = {
            'n_estimators': [100],
            'base_estimator': [DecisionTreeRegressor(max_depth = 19, random_state = 0)]
        }
grid_model = GridSearchCV(AdaBoostRegressor(random_state = 0), params, cv = 5)
grid_model.fit(X,y)


# In[291]:


# GRADIANT DESCENT
ada_sgd = grid_model.best_estimator_
ada_sgd.fit(X_train,y_train)
pred = ada_sgd.predict(X_test)


# In[293]:


SSE = np.sum((y_test-pred)**2) # ** kwargs is a common idiom to allow arbitrary number of arguments to functions
SST = np.sum((y_test-np.mean(pred))**2)
SSR = SST-SSE
R2 =1-(SSE/SST)


# In[295]:


print('Best Parameters :')
print(grid_model.best_params_)
print('')

print('Best Score      : %.2f'%grid_model.best_score_)
print('Test R^2        : %.2f'%grid_model.best_estimator_.score(X_test, y_test))
print('Training R^2    : %.2f'%grid_model.best_estimator_.score(X_train, y_train))
print('')
print('Model Fit:')
print('R2        :%.2f'%r2_score(y_test,pred)) #R^2
print('RMSE      :%.2f'%math.sqrt(mean_squared_error(y_test, pred))) #Root-Mean Square Error (RMSE)


# In[296]:


# Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestRegressor
RegModel = RandomForestRegressor(max_depth=10, n_estimators=100,criterion='mse')
# Good range for max_depth: 2-10 and n_estimators: 100-1000

# Printing all the parameters of Random Forest
print(RegModel)

# Creating the model on Training Data
RF=RegModel.fit(X_train,y_train)
prediction=RF.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(RF.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# In[297]:


print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['cnt']-TestingDataResults['Predictedcnt']))/TestingDataResults['cnt'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

MAPE=np.mean(TestingDataResults['APE'])
Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[298]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[299]:


#Graph k-fold score vs no. of estimators in Random Forest
scores = []
for i in range(10,101,10):
    scores.append(cross_val_score(RandomForestRegressor(n_estimators=i,random_state=9),
                                  X,y,cv=4).mean())


# In[300]:


plt.plot(range(10,101,10),scores)
plt.xlabel('No. of DTs in RandomForest')
plt.ylabel('K-fold scores')
plt.show()


# In[301]:


params = {
            'n_estimators': [100,140,115,130],
            'max_depth': [11,14,16]
        }
model = GridSearchCV(RandomForestRegressor(), params,cv=4)
model.fit(X,y)


# In[302]:


model.best_params_


# In[303]:


model.best_score_


# In[304]:


best_model = model.best_estimator_


# In[305]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=5)


# In[306]:


best_model.fit(X_train,y_train)


# In[307]:


best_model.score(X_test,y_test)


# In[308]:


cross_val_score(RandomForestRegressor(n_estimators=110,max_depth=14),X,y,cv=4)


# In[ ]:




Conclusion :  we can see that the accuracy of the model was increased from 31.8 in multi linear regression model to 99.02 in Random Forest model when we used Grid search to get the best model score with Max depth of 13 and number of estimator 120 
and we can say from the R2 that the variables in this dataset have postive linear corelations 
we also can say that Randomforest is good model for this dataset 
# In[ ]:


*******************Thank you *********Any Questions ****************************************************


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




