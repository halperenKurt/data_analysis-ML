from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

svc = SVC(kernel="linear",random_state=42)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print(classification_report(y_test, y_pred))



fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,5),
                        subplot_kw={"xticks":[],"yticks":[]})

for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary",interpolation="nearest")
    ax.set_title(digits.target[i])
    
    
plt.show()