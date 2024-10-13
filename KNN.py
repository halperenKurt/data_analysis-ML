# -*- coding: utf-8 -*-
#sklearn ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# (1)Veri setinin incelenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

#(2)Machine Learning modelinin seçilmesi-KNN sınıflandırı
#(3)Modelin train edilmesi

X=cancer.data
y=cancer.target
# train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
knn = KNeighborsClassifier(n_neighbors=9) #komşu parametresini unutma
knn.fit(X_train, y_train)

#(4)Sonuçların Değerlendirilmesi

y_pred=knn.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#(5)Hiperparametre Ayarlanması

accuracy_values=[]
k_values=[]

for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train) 
    y_pred=knn.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy)
    accuracy_values.append(accuracy)
    k_values.append(k)
    

plt.figure()
plt.plot(k_values,accuracy_values)
plt.title("Accuracy accordint to K value")
plt.xlabel("K value")
plt.ylabel("accuracy")
plt.xticks(k_values)
plt.grid(True)

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X=np.sort(5*np.random.rand(40,1),axis=0) #features
y=np.sin(X).ravel() #target

#plt.plot(X, y) grafiği birleştiriyor
#plot.scatter(X,y) grafiği noktalar halinde gösteriyor

#add noise 
y[::5]+=1*(0.5-np.random.rand(8))


T=np.linspace(0,5,500)[:,np.newaxis]

for i,weight in enumerate(["uniform","distance"]):
    knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
    y_pred=knn.fit(X, y).predict(T)
    
    plt.subplot(2,1,i+1)
    plt.scatter(X, y, color="green",label="data")
    plt.plot(T,y_pred,color="blue",label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights={}".format(weight))
    
plt.tight_layout()
plt.show()


























