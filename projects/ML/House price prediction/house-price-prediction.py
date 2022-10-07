import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import
import matplotlib.pyplot as plt
Data = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")
print(Data)
print(Data.isnull().sum())
c = Data.isnull().sum()
# plt.plot(c)
# plt.show()
# to see all data not the abbreviation only.
for i in range(0, 81):
    print(i, "==", c[i])
# all my data =  [1460 rows x 81 columns]
# i found columns null is
# 3 == 259
#  6 == 1369 ---> will delete
#  30 == 37
# 31 == 37
# 32 == 38
# 33 == 37
# 34 == 0
# 35 == 38
# 42 == 1
# 57 == 690   ---> will delete
# 58 == 81
# 59 == 81
# 60 == 81
# 63 == 81
# 64 == 81
# 72 == 1453   ---> will delete
# 73 == 1179   ---> will delete
# 74 == 1406   ---> will delete
# now  i whant to see names of this column.
# i will delete ID column becouse it not affect on my result
print(Data.columns[6], Data.columns[57], Data.columns[72], Data.columns[73], Data.columns[74])
# this name of columns which i will delete it Alley FireplaceQu PoolQC Fence MiscFeature + ID
Data = Data.drop(["Alley","FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Id"], axis=1)
print(Data)
# now i will comvert null value to mean value but first must do labelEncoder
data_object = Data.select_dtypes(include="object")
data_non_object = Data.select_dtypes(exclude="object")
encoder = LabelEncoder()
for i in range(data_object.shape[1]):
    data_object.iloc[:, i] = encoder.fit_transform(data_object.iloc[:, i])
print(data_object)
Data = pd.concat([data_object, data_non_object], axis=1)
print(Data)
#  print(Data.isnull().sum())# -->there is no any null value after label encoder.
# now I can do cleaning for data.
cleaner = SimpleImputer(missing_values=np.nan, strategy="mean")
cleaned_data = cleaner.fit_transform(Data)
new_data = pd.DataFrame(cleaned_data, columns=Data.columns)
print(new_data.iloc[:, 63])  # --> test LabelEncoder
# now all null data or 0 data converted to mean value
x_train = new_data.iloc[:, :-1]
y_train = new_data.iloc[:, -1]
print(x_train)
print(y_train)
model = LinearRegression()
model.fit(x_train, y_train)
print("accurecy of training = ", model.score(x_train, y_train))
#---------------------------- Decision tree --------------------------------
model_decision_tree = DecisionTreeRegressor()
model_decision_tree.fit(x_train,y_train)
print("accurecy of decision tree algorithm", model_decision_tree.score(x_train, y_train))
#---------------------------- Random forest --------------------------------
model_random_forest = RandomForestRegressor()
model_random_forest.fit(x_train,y_train)
print("accurecy of random forest algorithm", model_random_forest.score(x_train,y_train))
#---------------------------- neighbor  --------------------------------
model_neighbor = KNeighborsRegressor()
model_neighbor.fit(x_train,y_train)
print("accurecy of neighbor algorithm", model_neighbor.score(x_train, y_train))
