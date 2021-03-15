import tensorflow
import keras
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

data= pd.read_csv("student-mat.csv", sep = ";")

print(data.head())

data = data[["G1", "G2", "G3", "failures", "studytime"]]
predict =  "G3"

x = np.array(data.drop([predict], 1))

y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

best = 0
"""
for modelrange in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("GradesModel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""
pickle_in = open("GradesModel.pickle", "rb")

linear = pickle.load(pickle_in)
accuracy = linear.score(x_test, y_test)
print(accuracy)
print("Coef: ",linear.coef_)
print("Intercept: ", linear.intercept_)


preds = linear.predict(x_test)

for x in range(len(preds)):
    print(preds[x], x_test[x], y_test[x])
p = "studytime"
style.use("ggplot")

pyplot.scatter(data[p], data["G3"])

pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()