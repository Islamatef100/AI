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
