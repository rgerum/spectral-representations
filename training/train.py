import numpy as np
from tensorflow import keras
import tensorflow as tf

print(tf.version.VERSION)

from includes.logging import get_output_path
from includes.dimension_reg_layer import DimensionReg
from includes.callbacks import SaveHistory
from includes.attacks import get_attack_metrics

from typing import Literal


def main(
        # which dataset to use
        dataset: Literal['mnist', 'cifar10'] = "mnist",
        # which model to train
        model_type: Literal['shallow_mlp', 'deep_mlp', 'cnn'] = "shallow_mlp",
        # the strength of the regularisation (beta)
        reg_strength: float = 1.,
        # the target slope value (alpha_target)
        reg_target: float = 1.,
        method: str = "alpha",
        # which seed to run for repeating experiments
        repeat: int = 0,
        # how many epochs to run
        epochs: int = 1,
        # where to save the results
        output: str = "logs/test",
        # noise level
        noise: float = 0,
        # smallest index of spectrum to fit
        min_x: int = 0,
        # largest index of spectrum to fit
        max_x: int = -1,
):
    # set the seed depending on the repeat
    tf.random.set_seed((repeat + 1) * 1234)
    np.random.seed((repeat + 1) * 1234)

    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = getattr(keras.datasets, dataset).load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    # add a color channel if not present
    if len(x_train.shape) == 3:
        x_train = x_train[..., None]
        x_test = x_test[..., None]

    # get the number of classes
    num_classes = np.max(y_test) + 1
    # convert
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    cb = SaveHistory(get_output_path(main, locals()),
                     additional_logs_callback=[get_attack_metrics((x_test, y_test), np.arange(0, 0.2, 0.01))])

    if model_type == "shallow_mlp":
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),

            keras.layers.Flatten(),
            #tf.keras.layers.GaussianNoise(noise),
            keras.layers.Dense(units=2000, activation='tanh'),
            DimensionReg(reg_strength, reg_target, min_x=min_x, max_x=max_x),
            tf.keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        batch_size = 1500
    elif model_type == "deep_mlp":
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),

            keras.layers.Flatten(),
            keras.layers.Dense(units=1000, activation='tanh'),
            keras.layers.Dense(units=1000, activation='tanh'),
            keras.layers.Dense(units=1000, activation='tanh'),
            DimensionReg(reg_strength, reg_target, min_x=min_x, max_x=max_x),
            tf.keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        batch_size = 1500
    elif model_type == "cnn":
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),

            keras.layers.Conv2D(16, 3, activation='tanh'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32, 3, activation='tanh'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1000, activation='tanh'),
            DimensionReg(reg_strength, reg_target, min_x=min_x, max_x=max_x),
            tf.keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        batch_size = 6000
    else:
        raise ValueError("Unknown model type", model_type)
    if max_x == -1:
        max_x = batch_size // 2
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_test, y_test), callbacks=[cb])


if __name__ == "__main__":
    import fire

    fire.Fire(main)
