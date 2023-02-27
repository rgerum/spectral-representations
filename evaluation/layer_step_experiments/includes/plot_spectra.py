
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent))
from spectral_representations.regularizer.dimension_reg_layer import get_alpha
import yaml
import matplotlib.pyplot as plt
import pylustrator
#pylustrator.start()
import pandas as pd

def get_spectrum(folder, params, xmax, dataset):
    output_path = Path(folder) / "spectrum.npz"
    if output_path.exists():
        return np.load(output_path, allow_pickle=True)['arr_0'][()]
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

    # get the number of classes
    num_classes = np.max(y_test) + 1
    # convert
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    hidden = model2.predict(x_test[:batch_size])
    # alpha, mse, r2, x, y, (t, m, t2, m2) = get_alpha(hidden, min_x=min_x, max_x=max_x, target_alpha=reg_target)
    data = get_alpha(hidden, min_x=min_x, max_x=max_x, target_alpha=reg_target)

    np.savez(output_path, data)
    return data

def plot_folder(folder, xmax, dataset="mnist"):
    results_folder = Path(__file__).parent.parent.parent.parent / "training"
    folder = results_folder / folder
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
    #plt.plot(history.epoch, history.val_accuracy)
    #plt.ylim(0, 1)
    #return
    #return
    reg_target = params["reg_target"]
    min_x = 0
    max_x = xmax

    data = get_spectrum(folder, params, xmax, dataset)
    alpha = data["alpha"]
    mse = data["mse"]
    r2 = data["r2"]
    x, y = data["spectrum"]["x"], data["spectrum"]["y"]
    m2, t2 = data["target_line"]["m"], data["target_line"]["t"]

    print(f"*** *** {alpha=} {mse=} {r2=}")
    m, t = np.polyfit(x[min_x:max_x], y[min_x:max_x], deg=1)
    m2, t2 = m, t
    plt.title(f"{-m2:.2f}")
    plt.plot(10**x, 10**y, zorder=10)

    print("-----", m2, -reg_target, t2, t)
    #plt.axvline(10**x[min_x], color="k", lw=0.8)
    #plt.axvline(10**x[max_x], color="k", lw=0.8)
    plt.plot(10**x, 10**(m2 * x + t2), "k-", lw=1)
    plt.loglog([])
    plt.axvline(max_x, color='r', lw=0.5)
    plt.xlabel("index")
    plt.ylabel("variance")
    plt.ylim(1e-6, 1e0)
