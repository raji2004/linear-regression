import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import  style

data = pd.read_csv("breast-cancer-wisconsin.data", sep=',')
data = data[['out','time','a','b','c','e','f','g','h']]
pred = 'time'
x = np.array(data.drop([pred],1))

y = np.array(data[pred])
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1)
best = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    if acc > best:
        best= acc
        with open('time.pickle','wb') as f:
            pickle.dump(model, f)
        print(acc)

predict = model.predict(x_test)
for x in range(len(predict)):
    print(predict[x],x_test[x], y_test[x])
