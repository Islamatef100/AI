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
