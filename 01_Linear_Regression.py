import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("dataset/student-mat.csv", sep=";")

#print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

'''
best_score=0

for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)

    print("Accuracy %:", accuracy)

    if accuracy > best_score:
        best_score = accuracy
        with open("save_models/studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        print("New File Writen!!!")
'''

pickle_in = open("save_models/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    # x_test = "G1", "G2", "studytime", "failures", "absences"
    # y_test = G3
    print(predictions[x], x_test[x], y_test[x])

pick = 'G1' # "G1", "G2", "studytime", "failures", "absences"
style.use("ggplot")
pyplot.scatter(data[pick], data[predict])
pyplot.xlabel(pick)
pyplot.ylabel(predict)
pyplot.show()



