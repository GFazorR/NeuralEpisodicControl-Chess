from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.models import Sequential

import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    print(device_lib.list_local_devices())
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.test.is_gpu_available()
    model = Sequential()
    model.add(Conv2D(24, (2, 2), activation='relu', input_shape=(8, 8, 12)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(48, (2, 2), activation='linear'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())

    print(model.summary())
