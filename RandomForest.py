#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# veri setinin incelenmesi
oli = fetch_olivetti_faces()


X = oli.data
y= oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=43)

accuracy_values = []
i_values = []
for i in range(50,60):
    rf_clf = RandomForestClassifier(n_estimators=i,random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred=rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    i_values.append(i)



plt.figure()

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i],cmap="gray")
    plt.axis("off")
    
plt.show()


# %%
#Random Forest Regression
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

rf_regr = RandomForestRegressor(random_state=42)

rf_regr.fit(X_train, y_train)

y_pred = rf_regr.predict(X_test)

mse= mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
print("rmse:",rmse)



























