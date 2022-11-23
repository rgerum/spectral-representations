import tensorflow as tf
import numpy as np


def load_data():
    data_augmentation = True
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    return (x_train, y_train), (x_test, y_test)
