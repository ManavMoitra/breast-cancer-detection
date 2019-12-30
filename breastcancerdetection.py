#importing the libraries
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#loading the datasets
cancer=load_breast_cancer()
df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
df["Melignant or Beningn"]=pd.Categorical.from_codes(cancer.target,cancer.target_names)

df["Melignant or Beningn"]=pd.factorize(df["Melignant or Beningn"])[0]

#selecting the features and outputs
X=df.iloc[:,:-1]
y=df.iloc[:,30]

#splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)

#scaling of the data
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#dimensionality reduction using principal component analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit_transform(X_train)
pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

#using of K Nearest Neighbors Algorithm for classification
from sklearn.neighbors import KNeighborsClassifier

classifier1=KNeighborsClassifier(n_neighbors=4,metric="minkowski",p=2)
classifier1.fit(X_train,y_train)
y_pred1=classifier1.predict(X_test)

#searching of suitable parameters using gridsearch
from sklearn.model_selection import GridSearchCV
Parameters=[{"n_neighbors" : [1,2,3,4,5,6,7,8],"metric": ["minkowski"]},
            {"n_neighbors" : [1,2,3,4,5,6,7,8],"metric": ["manhattan"]}]
grid_search=GridSearchCV(estimator=classifier1,
                         param_grid=Parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_
print(best_parameters)

#predicting the accuracy of the model
from sklearn.metrics import accuracy_score
print("accuracy score for KNeighbors Classifier is ",accuracy_score(y_test,y_pred1))

#jaccard index
from sklearn.metrics import jaccard_score
print("Jaccard score using KNeighbors Classifier is ",jaccard_score(y_test,y_pred1))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)
print(cm1)

from sklearn.svm import SVC

classifier2=SVC(kernel='linear',C=1,gamma=0.1)
classifier2.fit(X_train,y_train)
y_pred2=classifier2.predict(X_test)

parameters_svm=[{"kernel" : ["rbf"],"C" : [0.01,0.1,1,2,10,100,1000],'gamma':[0.1,0.2,0.4,1,2,1.4,2,0.6]},                 
                {"kernel" : ["linear"],"C":[0.01,1,2,10,100,1000],'gamma':[0.1,0.2,0.4,0,1,2,1.4,2,2.6]}]
grid_search=GridSearchCV(estimator=classifier2,
                         param_grid=parameters_svm,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_score=grid_search.best_score_
                         
print(best_parameters)

print(accuracy_score(y_test,y_pred2))



        
from sklearn.tree import DecisionTreeClassifier
classifier3= DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=0)
classifier3.fit(X_train,y_train)

y_pred3=classifier3.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred3))

from sklearn.ensemble import RandomForestClassifier

classifier4=RandomForestClassifier(n_estimators=100,max_depth=5,criterion="entropy",random_state=0)
classifier4.fit(X_train,y_train)
y_pred4=classifier4.predict(X_test)
print(accuracy_score(y_test,y_pred4))

from sklearn.ensemble import GradientBoostingClassifier
classifier5= GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=1,random_state=0)

classifier5.fit(X_train,y_train)
y_pred5=classifier5.predict(X_test)
print(accuracy_score(y_test,y_pred5))
