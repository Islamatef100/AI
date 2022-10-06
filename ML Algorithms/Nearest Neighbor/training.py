#-----------nearest neighbor---------------------
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#--------------- for classification------------------------------
Data = pd.read_csv("Iris.csv")
x = Data.iloc[:, :-1]
y = Data.iloc[:, -1]
x_train, x_test, y_train, y_test  = train_test_split(x,y,train_size=.7)
# inpute is defualt
#  n_neighbors = (...) -->number of neighbors
Model = KNeighborsClassifier()
Model.fit(x_train, y_train)
print(Model.score(x_train, y_train))
print(Model.score(x_test, y_test))
#-------------------for regresion--------------------------------
from sklearn.neighbors import KNeighborsRegressor
data = pd.read_csv("train_regression.csv")
data = data.dropna()
xx = data.iloc[:, :-1]
yy = data.iloc[:, -1]
xx_train, xx_test, yy_train, yy_test  = train_test_split(xx,yy,train_size=.7)
# inpute is defualt
#  n_neighbors = (...) -->number of neighbors
model = KNeighborsRegressor()
model.fit(xx_train, yy_train)
print(model.score(xx_train, yy_train))
print(model.score(xx_test, yy_test))
