from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

diabetes= load_diabetes()
df=pd.DataFrame(data=diabetes.data,columns=diabetes.feature_names)
df["target"]=diabetes.target

X=diabetes.data
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

knn=KNeighborsRegressor(n_neighbors=5)
y_pred=knn.fit(X_train, y_train).predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek ve Tahmin Edilen Değerler Arasındaki İlişki')
plt.show()