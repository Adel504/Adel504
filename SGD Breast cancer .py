#!/usr/bin/env python
# coding: utf-8

# In[30]:


#import dataset
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(True) 


# In[31]:


#scaling data is necessary for making gradient descent faster 
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = pd.DataFrame(sc.fit_transform(X))
y = pd.Series(y)


# In[32]:


y.head()


# In[33]:


X_scaled.head()


# In[34]:


#Gradient descent can used in different models depending 
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='hinge', learning_rate='constant',eta0=0.5)
perc = SGDClassifier(loss='perceptron', learning_rate='constant',eta0=0.5)
logreg = SGDClassifier(loss='log', learning_rate='constant',eta0=0.5)


# In[37]:


#Gradient descent can used in different models depending 
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='log', learning_rate='constant',eta0=1.0)
perc = SGDClassifier(loss='perceptron', learning_rate='optimal',eta0=1.0)
logreg = SGDClassifier(loss='log', learning_rate='constant',eta0=0.5)


# In[6]:


from sklearn.model_selection import cross_val_score
print("svm's 4-fold score:",cross_val_score(svm,X_scaled,y,cv=4).mean())
print("perceptron's 4-fold score:",cross_val_score(perc,X_scaled,y,cv=4).mean())
print("logistic regression's 4-fold score:",cross_val_score(logreg,X_scaled,y,cv=4).mean())


# In[39]:


from sklearn.model_selection import cross_val_score
print("svm's 4-fold score:",cross_val_score(svm,X_scaled,y,cv=4).mean())
print("perceptron's 4-fold score:",cross_val_score(perc,X_scaled,y,cv=5).mean())
print("logistic regression's 4-fold score:",cross_val_score(logreg,X_scaled,y,cv=4).mean())


# In[36]:


from sklearn.model_selection import cross_val_score
print("svm's 4-fold score:",cross_val_score(svm,X_scaled,y,cv=4).mean())
print("perceptron's 4-fold score:",cross_val_score(perc,X_scaled,y,cv=4).mean())
print("logistic regression's 4-fold score:",cross_val_score(logreg,X_scaled,y,cv=4).mean())


# In[40]:


from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import matplotlib.pyplot as plt


# In[41]:


model A: train score = 99%, val score = 89%

model B: train score = 80%, val score = 79%


# In[56]:



from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import matplotlib.pyplot as plt


# In[57]:


#import dataset
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(True) 


# In[58]:



breast_cancer = load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)


# In[59]:


abreg = AdaBoostRegressor()


# In[60]:


params = {
 'n_estimators': [50, 100],
 'learning_rate' : [0.01, 0.05, 0.1, 0.5],
 'loss' : ['linear', 'square', 'exponential']
 }


# In[61]:


score = make_scorer(mean_squared_error)


# In[62]:


gridsearch=GridSearchCV(abreg, params, cv=5, return_train_score=True)
gridsearch.fit(x, y)
GridSearchCV(cv=5, error_score='raise',
       estimator=AdaBoostRegressor(base_estimator=None, learning_rate=1.0, 
       loss='linear', n_estimators=50, random_state=None),
        iid=True, n_jobs=1,
       param_grid={'n_estimators': [50, 100], 
                   'learning_rate': [0.01, 0.05, 0.1, 0.5], 
                   'loss': ['linear', 'square', 'exponential']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring=None, verbose=0) 


# In[63]:


gridsearch=GridSearchCV(abreg,params,scoring=score,cv=5,return_train_score=True)


# In[65]:


gridsearch=GridSearchCV(abreg,params,scoring=score,cv=5,return_train_score=True)


# In[66]:



print(gridsearch.best_params_)
{'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 50}
print(gridsearch.best_score_)


# In[54]:


print(gridsearch.best_params_)
{'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 50}
print(gridsearch.best_score_)


# In[55]:


best_estim=gridsearch.best_estimator_
print(best_estim)
AdaBoostRegressor(base_estimator=None, learning_rate=0.5, loss='exponential',
         n_estimators=50, random_state=None)


# In[55]:


ytr_pred=best_estim.predict(xtrain)
mse = mean_squared_error(ytr_pred,ytrain)
r2 = r2_score(ytr_pred,ytrain)
print("MSE: %.2f" % mse)
MSE: 7.54
print("R2: %.2f" % r2)
R2: 0.89


# In[56]:


ypred=best_estim.predict(xtest)
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
print("MSE: %.2f" % mse)
MSE: 11.51
print("R2: %.2f" % r2)
R2: 0.85 


# In[57]:


x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[67]:


from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import matplotlib.pyplot as plt


# In[68]:


breast_cancer = load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)


# In[69]:


abreg = AdaBoostRegressor()


# In[70]:


abreg = AdaBoostRegressor()
params = {
 'n_estimators': [50, 100],
 'learning_rate' : [0.01, 0.05, 0.1, 0.5],
 'loss' : ['linear', 'square', 'exponential']
 }


# In[71]:



score = make_scorer(mean_squared_error)


# In[72]:


gridsearch = GridSearchCV(abreg, params, cv=5, return_train_score=True)
gridsearch.fit(xtrain, ytrain)
print(gridsearch.best_params_)


# In[73]:


best_estim=gridsearch.best_estimator_
print(best_estim)


# In[74]:


best_estim.fit(xtrain,ytrain)


# In[75]:


ytr_pred=best_estim.predict(xtrain)
mse = mean_squared_error(ytr_pred,ytrain)
r2 = r2_score(ytr_pred,ytrain)
print("MSE: %.2f" % mse)
print("R2: %.2f" % r2)


# In[76]:


ypred=best_estim.predict(xtest)
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
print("MSE: %.2f" % mse)
print("R2: %.2f" % r2)


# In[77]:


x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
 


# In[78]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# For plotting the classification results
from mlxtend.plotting import plot_decision_regions


# In[79]:


# Importing the dataset
dataset = load_breast_cancer() 
# Converting to pandas DataFrame
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)
df.head()


# In[80]:



print("Total samples in our dataset is: {}".format(df.shape[0]))


# In[81]:


df.describe()


# In[82]:


# Selecting the features
features = ['mean perimeter', 'mean texture']
x = df[features]
# Target Variable
y = df['target']


# In[83]:


# Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 25 )


# In[84]:


# Fitting SGD Classifier to the Training set
model = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
model.fit(x_train, y_train)


# In[85]:


# Predicting the results
y_pred = model.predict(x_test)


# In[86]:


# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)
# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('SGD Classifier Accuracy of the model: {:.2f}%'.format(accuracy*100))


# In[87]:


# Plotting the decision boundary
plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
plt.title("Decision boundary using SGD Classifier (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:


X = X.values


# In[ ]:





# In[ ]:




