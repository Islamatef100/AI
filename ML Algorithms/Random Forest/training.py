# ---------------for classification-------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
# for classification
from sklearn.ensemble import RandomForestClassifier
Data = pd.read_csv("Iris.csv")
x = Data.iloc[:, :-1]
y = Data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y)
#  i will use use default impute
#  n_estimators=()--->number of forest
#  max_depth=()-->depth of tree
model_classifier = RandomForestClassifier()
model_classifier.fit(x_train, y_train)
print(model_classifier.score(x_train, y_train))
print(model_classifier.score(x_test, y_test))
 # what abotu regresssion??
from sklearn.ensemble import RandomForestRegressor
Data2 = pd.read_csv("train_regression.csv")
# i should to remove or operate with null data
Data2 = Data2.dropna()
xx = Data2.iloc[:, :-1]
yy = Data2.iloc[:, -1]
xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy)
#  i will use use default impute
#  n_estimators=()--->number of forest
#  max_depth=()-->depth of tree
model_regressor = RandomForestRegressor()
model_regressor.fit(xx_train, yy_train)
print(model_regressor.score(xx_train, yy_train))
print(model_regressor.score(xx_test, yy_test))





















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
