import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import  style


data = pd.read_csv('student-mat.csv', sep=';')


data = data[['G1','G2','G3','studytime','failures','absences']]
predict =  'G3'
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test,y_train,y_test= sklearn.model_selection.train_test_split(x,y, test_size=0.1)
# best = 0
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train,y_train)
#     acc = linear.score(x_test,y_test)
#     print(acc)
#     if acc > best:
#         best= acc
#         with open('studentmodel.pickle','wb') as f:
#             pickle.dump(linear, f)
pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# print("co: " , linear.coef_)
# print("intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x], y_test[x])

p='G2'
style.use('ggplot')
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel('first score')
pyplot.ylabel('final score')
pyplot.show()