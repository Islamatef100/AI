"""
In this section we will do the "Hello World" of deep learning: training a deep learning model to correctly classify hand-written digits.
"""

from tensorflow.keras.datasets import mnist

# the data, split between train and validation sets
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

"""Exploring the MNIST Data"""

x_train.shape

x_valid.shape

x_train.dtype
x_train.min()

x_train.max()

x_train[0]

"""to see one image from my data"""

import matplotlib.pyplot as plt

image = x_train[0]
plt.imshow(image, cmap='gray')

"""to see the label name"""

y_train[0]

"""the inpute for neural network is 28*28*1 -->pixel so i will make the inpute 784 == 28*28*1 """

x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)

"""to make sure from my steps"""

x_train.shape

x_train[0]

"""make the image binary because i don't need gray or rgb image and it will make the processing easier"""

x_train = x_train / 255
x_valid = x_valid / 255

x_train.dtype

x_train.min()

x_train.max()

"""here i want make y not identical value but like vector of 0 and 1 for corect label if dataset is have vector of label i don;t need do that."""

import tensorflow.keras as keras
num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

y_train[0:9]

"""creat neural network"""

from tensorflow.keras.models import Sequential
model = Sequential()

"""here will start impute layr"""

from tensorflow.keras.layers import Dense

"""input layer

Dense  -- >i add layer
activiation --> function operate with output of layer and make in suitable for next layer
units -->number of neurons
input_shape()--> this layer for input and the shap of input data is..
"""

model.add(Dense(units=512, activation='relu', input_shape=(784,)))

"""Creating the Hidden Layer

dens --> layer
units -->number of neurons
"""

model.add(Dense(units = 512, activation='relu'))

"""Creating the Output Layer

softmax --.activationfunction -->now output is probability for all label and this take max value and use it as the prediction
units =10 -->because i have 10 label only and output is 10 values and softmax make all value one value,
"""

model.add(Dense(units = 10, activation='softmax'))

"""s

Summarizing the Model
"""

model.summary()

"""Compiling the Model

loss --> the way for Calculate the accurecy
matrics --> to show accurecy in training.
"""

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

"""Training the Model

i use train and test in the same line becouse try test at all epoch not in final result for weights.
"""

history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)

"""to clear gpu."""

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
