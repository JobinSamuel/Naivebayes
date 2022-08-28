import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import accuracy_score

###Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train and test datasets are given separately. Use both for model building. 

sal_tr = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/SalaryData_Train.csv")
sal_ts = pd.read_csv("/Users/jobinsamuel/Desktop/Assignments/Naive bias/Datasets_Naive Bayes/SalaryData_Test.csv")

sal_ts.head()
sal_tr.head()
sal_ts.dtypes
sal_tr = pd.get_dummies(sal_tr,columns=['workclass','native','sex','race','relationship'],drop_first=True)
sal_ts = pd.get_dummies(sal_ts,columns=['Salary'],drop_first = True)
sal_ts.head()
sal_ts.rename(columns = {'Salary_ >50K':'Sal'}, inplace = True)
X = sal_tr.drop(columns = ['Salary','maritalstatus','education','occupation'])
Y = sal_ts.Sal.values
X = X.iloc[0:15060,]
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.30,random_state = 0) 


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(Xtrain, Ytrain)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(Xtest)
accuracy_test_m = np.mean(test_pred_m == Ytest)
accuracy_test_m

# Training Data accuracy
train_pred_m = classifier_mb.predict(Xtrain)
accuracy_train_m = np.mean(train_pred_m == Ytrain)
accuracy_train_m


#Model -2
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(Xtrain, Ytrain)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(Xtest)
accuracy_test_lap = np.mean(test_pred_lap == Ytest)
accuracy_test_lap


# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(Xtrain)
accuracy_train_lap = np.mean(train_pred_lap == Ytrain)
accuracy_train_lap

#Model -3
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(Xtrain,Ytrain)
predic_y = model.predict(Xtest)

accuracy_score = accuracy_score(Ytest, predic_y) 
accuracy_score
