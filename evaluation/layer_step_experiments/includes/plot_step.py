import tensorflow as tf
import numpy as np
import glob
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from spectral_representations.regularizer.dimension_reg_layer import get_alpha
import yaml
import matplotlib.pyplot as plt
import pylustrator
#pylustrator.start()
import pandas as pd


def get_step(folder, xmax, dataset):
    with open(Path(folder).parent / "arguments.yaml", "r") as stream:
        params = yaml.safe_load(stream)

    reg_target = params["reg_target"]
    min_x = 0  # params["min_x"]
    max_x = xmax
    # max_x = 4#params["max_x"]

    batch_size = 1500
    if max_x == -1:
        max_x = batch_size // 2

    model = tf.keras.models.load_model(folder)

    model2 = tf.keras.models.Sequential(
        [tf.keras.Input(model.input.shape[1:])] + model.layers[:-1],
    )
    model2.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, dataset).load_data()
    if len(y_train.shape) == 2:
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    indices = y_train < xmax
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = y_test < xmax
    x_test = x_test[indices]
    y_test = y_test[indices]


    hidden = model2.predict(x_test[:1500])
    data = get_alpha(hidden, min_x=min_x, max_x=max_x, target_alpha=reg_target)
    x, y = data["spectrum"]["x"], data["spectrum"]["y"]

    return y[xmax-1] - y[xmax+1]

index = 0
def plot_folder(folder, xmax, dataset="mnist"):
    global index
    try:
        with open(Path(folder).parent / "arguments.yaml", "r") as stream:
            params = yaml.safe_load(stream)
    except FileNotFoundError:
        return
        folder = Path(folder)
        print(os.getcwd())
        for i in range(10):
            print(folder.exists(), folder)
            folder = folder.parent
        raise
    history = pd.read_csv(Path(folder).parent / "data.csv")
    ratio = get_step(folder, xmax, dataset)
    plt.subplot(131)
    plt.plot(history.val_accuracy.iloc[-1], ratio, "o")
    plt.xlabel("accuracy")
    plt.ylabel("spectral step")
    plt.subplot(132)
    plt.plot(index, ratio, "o")
    plt.xlabel("depth")
    plt.ylabel("spectral step")
    plt.subplot(133)
    plt.plot(index, history.val_accuracy.iloc[-1], "o")
    plt.xlabel("depth")
    plt.ylabel("accuracy")