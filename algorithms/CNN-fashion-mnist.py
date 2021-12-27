import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

x_train = train_images.astype('float32') / 255
x_test = test_images.astype('float32') / 255

model = tf.keras.Sequential([
  layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)),
  layers.Conv2D(8, 5, strides=2, activation='relu'),
  layers.ZeroPadding2D(padding=1),
  layers.Conv2D(16, 3, padding='valid',  strides=2, activation='relu'),
  layers.ZeroPadding2D(padding=1),
  layers.Conv2D(32, 3, padding='valid',  strides=2, activation='relu'),
  layers.ZeroPadding2D(padding=1),
  layers.Conv2D(32, 3, padding='valid',  strides=2, activation='relu'),
  layers.AveragePooling2D(1),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(10),
  layers.Softmax()
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train,
         train_labels,
         batch_size=64,
         epochs=10,
         validation_data=(x_test, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
