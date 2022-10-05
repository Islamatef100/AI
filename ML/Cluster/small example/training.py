
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv("Iris.csv")
# i don not want output data
data = data.drop("label", axis=1)
print(data)
number_of_cluster = []
j = []
# to determine number of cluster
for i in range(1,11):
    model = KMeans(n_clusters=i)
    model.fit(data)
    number_of_cluster.append(i)
    #  --> model.inertia_ --> this retutn tatio of error.
    j.append(model.inertia_)
print(number_of_cluster)
print(j)
plt.plot(number_of_cluster, j)
plt.show()
# now i know best number of cluster is (4)
