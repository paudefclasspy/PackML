import tensorflow
import keras
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import style

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
classCar = le.fit_transform(list(data["classCar"]))


predict = "classCar"

x = list(zip(buying,maint,doors,persons,lug_boot,safety))
y = list(classCar)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)


model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)
predicted = model.predict(x_test)
names = ["unacc","acc","good", "very good"]

for x in range(len(predicted)):
    print("Predicted: ",names[predicted[x]], "Data: ", x_test[x], "Actual: ",names[y_test[x]])
    n = model.kneighbors([x_test[x]],n_neighbors = 9, return_distance = True)
    print(n)