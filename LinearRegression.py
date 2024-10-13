from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#veri seti oluştur
X = np.random.rand(100,1)

# y = 3 + 4X
y = 3 + 4 * X + 20*np.random.rand(100,1)


lin_reg = LinearRegression()
lin_reg.fit(X, y)


plt.figure()
plt.scatter(X, y)
plt.plot(X,lin_reg.predict(X),color="red",alpha=0.7)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")

# y = 3 + 4X -> a0+a1x

a1 = lin_reg.coef_[0][0] 
print("a1: ",a1)
a0 = lin_reg.intercept_[0]
print("a0: ",a0)

for i in range(100):
    y_ = a0+a1*X
    plt.plot(X, y_,color="green",alpha=0.7)

#%%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
diabetes = load_diabetes()
diabetes_X,diabetes_y = load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:,np.newaxis,2]


X = diabetes.data
y = diabetes.target

X_train,X_test,y_train,y_test=train_test_split(diabetes_X,y,test_size=0.3,random_state=42)


lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("mse: ",mse)

r2 = r2_score(y_test, y_pred)
print("r2: ",r2)

plt.figure()
plt.scatter(X_test, y_test,color="black")
plt.plot(X_test, y_pred,color="green")


































