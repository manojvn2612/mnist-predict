import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_file = pd.read_csv("mnist_train_small.csv",header = None)
test_file = pd.read_csv("mnist_test.csv",header = None)
y_train = train_file.pop(0)
x_train = train_file
y_test = test_file.pop(0)
x_test = test_file

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation = 'softmax'))
model.compile(optimizer = 'Adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 4)

loss,ac= model.evaluate(x_test,y_test)
print(loss,ac)

model.save("Mnist_training.model")
load_model = models.load_model("Mnist_training.model")

predict = load_model.predict([x_test])
np.argmax(predict[1])

plt.imshow(x_test[0],cmap='Greys')
