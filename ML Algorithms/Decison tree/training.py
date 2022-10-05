import pandas as pd
# desion tree for classification
from sklearn.tree import DecisionTreeClassifier
# desion tree for regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#  ->  use desion tree for classification

Data = pd.read_csv("Iris.csv")
print(Data.head)
x = Data.iloc[:, :-1]
y = Data.iloc[:, -1]
x_trin, x_test, y_trin, y_test = train_test_split(x, y, train_size=0.7)
model = DecisionTreeClassifier()
# max_depth= ()  -> for tree
model.fit(x_trin, y_trin)
print(model.score(x_trin, y_trin))
print(model.score(x_test, y_test))
#  --> us desion tree   for regression


data2 = pd.read_csv("train_regression.csv")
data2 = data2.dropna()
x2 = data2.iloc[:, :-1]
y2 = data2.iloc[:, -1]
x2_trin, x2_test, y2_trin, y2_test = train_test_split(x2, y2, train_size=0.7)
model2 = DecisionTreeRegressor()
model2.fit(x2_trin, y2_trin)
print(model2.score(x2_trin, y2_trin))
print(model2.score(x2_test, y2_test))














# import pandas as pd
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
#     #  --> model.inertia_ --> this retutn tatio of error.
#     j.append(model.inertia_)
# print(number_of_cluster)
# print(j)
# plt.plot(number_of_cluster, j)
# plt.show()
# # now i know best number of cluster is (4)


























# import pandas as pd
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
#     #  --> model.inertia_ --> this retutn tatio of error.
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
