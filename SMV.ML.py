import sklearn
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

print(x_train, y_train)

classes = ["setosa", "versicolor", "virginica"]