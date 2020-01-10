import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("dataset/car.data")
print(data.head())

le = preprocessing.LabelEncoder()
# buying,maint,door,persons,lug_boot,safety,class
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("Accuracy % ", acc)

names = ["unacc", "acc", "good", "vgood"]
predicted = model.predict(x_test)

for x in range(len(x_test)):
    print("Pediction: ", predicted[x], "Data: ", x_test[x], "Actual: ", y_test[x])

    n = model.kneighbors([x_test[x]], 5, True)
    print("N:", n)
    print("=============")
