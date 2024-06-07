import pandas as pd

import numpy as np

import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib import style

import pickle

#read data
data = pd.read_csv("/home/user/guess-the-score/Car_classification/car.data")

#label encoding
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#set x and y
X  = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

#split data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

best = 0
for _ in range(1000000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    #Create KNN Classifier
    model = KNeighborsClassifier(n_neighbors=9)

    #train the model using the training sets
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    
    if acc > best:
        best = acc
        
        #save the model to disk
        with open("Car_class.pickle", "wb") as f:
            pickle.dump(model, f)
print(best)

#load the model
pickle_in = open("Car_class.pickle", "rb")
model = pickle.load(pickle_in)

#make predictions on the testing set
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

#print results
for x in range(len(x_test)):
    print("predicted: ", names[predicted[x]], "data: ", x_test[x], "actual: ", names[y_test[x]])
