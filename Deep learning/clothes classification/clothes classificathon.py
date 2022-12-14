import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
#  split data to training and testing data
(x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()
# list of all clothes label.
clothes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle"]
#  test data
plt.figure()  # to make a place for a picture
plt.imshow(x_train[0], cmap="gray")
plt.colorbar()  # for drawing colum par beside the image.
plt.show()
# to make operations faster.
x_test = x_test/255.0
x_train = x_train/255.0
# create model
model = keras.models.Sequential()
model.add(keras.layers.Flatten())  # input layer
model.add(keras.layers.Dense(256, activation=tf.nn.relu))# hidden layer
model.add(keras.layers.Dropout(0.0))  # for drop
model.add(keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train ,y_train, epochs=10)  # training
test_loss, test_acc=model.evaluate(x_test, y_test)
print(test_loss, test_acc)
model.save("clothes_classification.model")
new_model=keras.models.load_model("clothes_classification.model")
