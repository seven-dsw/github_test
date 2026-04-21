import numpy as np
import pandas as pd
import seaborn as sns
from distributed import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

churn_pd = pd.read_csv('churn.csv')

churn_pd=pd.get_dummies(churn_pd)
churn_pd.drop(['Churn_No','gender_Male'], axis=1, inplace=True)
churn_pd.rename(columns={'Churn_Yes':'flag'},inplace=True)

x=churn_pd[['Contract_Month','internet_other','PaymentElectronic']]
y=churn_pd['flag']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
estimator=LogisticRegression()
estimator.fit(x_train,y_train)
y_pred=estimator.predict(x_test)

my_accuracy_score=accuracy_score(y_test,y_pred)
print('my_accuracy_score=',my_accuracy_score)
my_score = estimator.score(x_test,y_test)
print('my_score=',my_score)

my_roc_auc_score = roc_auc_score(y_test,y_pred)
print('my_roc_auc_score=',my_roc_auc_score)
result = classification_report(y_test,y_pred,target_names=['flag0','flag1'])
print('result=',result)