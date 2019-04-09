"""
a Convolution Neural Network for classifying The MNIST dataset
the model performes very well it was reaching 99.26 accuracy at 10 epochs

The over all results:
                      precision    recall  f1-score   support

           0       1.00      0.97      0.99       980
           1       1.00      1.00      1.00      1135
           2       0.98      0.97      0.98      1032
           3       0.98      0.99      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.98      0.99      0.99       892
           6       0.99      0.99      0.99       958
           7       0.97      0.99      0.98      1028
           8       1.00      0.98      0.99       974
           9       0.99      0.98      0.99      1009

micro avg          0.99      0.99      0.99     10000
macro avg          0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
samples avg        0.99      0.99      0.99     10000


"""
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, AvgPool2D
from sklearn.metrics import classification_report
from keras.utils import to_categorical

"""EXTARACTING DATA"""

mnist = input_data.read_data_sets('sample_data/', one_hot=True)

train = mnist.train.images

test = mnist.test.images

def reshape_images(images):
    reshaped = []
    for image in images:
        reshaped.append(image.reshape(28,28))
    return np.array(reshaped)

train = reshape_images(train)
train = train.reshape((-1,28,28,1))
test = reshape_images(test)
test = test.reshape((-1,28,28,1))
train_labels = mnist.train.labels
test_labels =  mnist.test.labels
# view the first 5 images
for image in reshaped[:5]:
    plt.imshow(image, cmap='gray')
    plt.show()
    print('\n'*2)

# building the model

model = Sequential([
    Conv2D(128,kernel_size=(3,3), kernel_initializer='uniform',activation='relu',input_shape=(28,28,1)),
    MaxPool2D(),
    Conv2D(128,kernel_size=(3,3), kernel_initializer='uniform',activation='relu'),
    MaxPool2D(),
    Conv2D(64,kernel_size=(3,3), kernel_initializer='uniform',activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(512, activation='relu', kernel_initializer='uniform'),
    Dense(256, activation='relu', kernel_initializer='uniform'),
    Dense(10, activation='sigmoid', kernel_initializer='uniform'),
    
])

model.summary()

model.compile('adam','categorical_crossentropy',['accuracy'])

history = model.fit(train, train_labels, epochs=10, batch_size=32 ) # you could use history to do more visualization.

""" Evaluation """

prediction = model.predict_classes(test)
print(classification_report(test_labels, to_categorical(prediction, 10)))
