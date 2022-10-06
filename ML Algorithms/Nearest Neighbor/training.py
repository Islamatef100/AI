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




















#import pandas as pd
# from sklearn.model_selection  import train_test_split
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# data = pd.read_csv("Iris.csv")
# # i don not want output data
# data = data.drop("label", axis=1)
# print(data)
# number_of_cluster = []
# j = []
# # to determine number of cluster
# for i in range(1,11):
#     model = KMeans(n_clusters=i)
#     model.fit(data)
#     number_of_cluster.append(i)
#     #  --> model.inertia_ --> this return ratio of error.
#     j.append(model.inertia_)
# print(number_of_cluster)
# print(j)
# plt.plot(number_of_cluster, j)
# plt.show()
# # now i know best number of cluster is (4)




#-----------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# # for classification
# # to do label encoding
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPRegressor
# Data = pd.read_csv("train_regression.csv")
# Data = Data.dropna()
# print(Data.head())
# x = Data.iloc[:, :-1]
# y = Data.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=.7)
# # use default parameters->max-iter/activation/hidden_layer_sizes.
# model = MLPRegressor()
# model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))
