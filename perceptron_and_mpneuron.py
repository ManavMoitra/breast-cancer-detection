#IMPLEMENTATION OF MPNEURON  AND PERCEPTRON ON BREASTCANCER DATASET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import r2_score
#Getting the Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['class']=cancer.target
X=df.drop('class',axis=1)
y=df['class']

#Spltting The Data intoy Train and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

plt.plot(X_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()

X_binarised_train=X_train.apply(pd.cut,bins=2,labels=[1,0])

X_binarised_test=X_test.apply(pd.cut,bins=2,labels=[1,0])

X_binarised_train=X_binarised_train.values
X_binarised_test=X_binarised_test.values
#Creating A Class MPNEURON
class MPNEURON:
    def __init__(self):
        self.b=None
    
    def model(self,x):
        return(np.sum(x)>=self.b)
    def predict(self,X):
        
        Y=[]
        for x in X:
            result=self.model(x)
            Y.append(result)
        return(np.array(Y))
    
    def fit(self,X,Y):
        accuracy={}
        for b in range(X.shape[1]+1):
            self.b=b
            y_pred=self.predict(X)
            accuracy[b]=accuracy_score(y_pred,Y)
        best_b=max(accuracy,key=accuracy.get)
        print("optmal Value Of b is ",best_b)
        print("highest accuracy of b is ",accuracy[best_b])
mp_neuron=MPNEURON()
mp_neuron.fit(X_binarised_train,y_train)
y_pred1=mp_neuron.predict(X_binarised_test)
#Accuracy score is given as 
print("accuracy score using mp_neuron is given as ",accuracy_score(y_test,y_pred1))
#Jaccard score is given as
print("Jaccard Score is given as ",jaccard_score(y_test,y_pred1))
#confusion matrix is given as 
cm=confusion_matrix(y_test,y_pred1)
print(cm)
print(r2_score(y_test,y_pred1))

#Using of Perceptron Class
X_train=X_train.values
X_test=X_test.values
#PERCEPTRON CLASS    
class Perceptron:
    def __init__(self):
        self.b=None
        self.w=None
    
    def model(self,x):
        return 1 if np.dot(self.w,x)>=self.b else 0
    
    def predict(self,X):
        Y=[]
        for x in X:
            result=self.model(x)
            Y.append(result)
        return(np.array(Y))
        
    def fit(self,X,Y,epoch,lr):
        
        self.b=0
        self.w=np.ones(X.shape[1])
        accuracy={}
        max_accuracy=0
        for i in range(epoch):
            for x,y in zip(X,Y):
                
                y_pred=self.model(x)
                if y==1 and y_pred==0:
                    
                    self.w=self.w+lr*x
                    self.b=self.b+lr*1
                if y==0 and y_pred==1:
                    self.w=self.w-lr*x
                    self.b=self.b-lr*1
                    
                    
                accuracy[i]=accuracy_score(self.predict(X),Y)
                if(accuracy[i]>max_accuracy):                    
                    max_accuracy=accuracy[i]
                    chckptb=self.b
                    chckptw=self.w
                
        self.b=chckptb
        self.w=chckptw
        print("maximum accuracy is given as ",max_accuracy)
        print("Optimal Value of b is ",chckptb)
        print("Optimal Value of w is ",chckptw)
perceptron=Perceptron()
perceptron.fit(X_train,y_train,100,0.1)
y_pred2=perceptron.predict(X_test)  
#Accuracy For the Perceptron model
print("Accuracy score for perceptron model is ",accuracy_score(y_test,y_pred2))  
#Jaccard Score
print("Jaccard score using perceptron is found to be ",jaccard_score(y_test,y_pred2)) 
#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred2))
#r2 score can be given as 
print("R2 score for the perceptron model is ",r2_score(y_test,y_pred2))
plt.plot(perceptron.w)

            