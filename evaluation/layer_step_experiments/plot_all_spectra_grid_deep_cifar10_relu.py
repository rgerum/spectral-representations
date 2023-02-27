import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from includes.plot_spectra import plot_folder


def plot_rows(dataset="MNIST", mode="tanh"):
    for index in range(1,11):
        plt.subplot(3, 10, index)
        folder = f"multi_layer/dnn/{dataset}/{mode}/{index}_2_{dataset}_{mode}/model_save"
        plot_folder(folder, xmax=2, dataset=dataset.lower())

    for index in range(1,11):
        plt.subplot(3, 10, index+10)
        folder = f"multi_layer/dnn/{dataset}/{mode}/{index}_5_{dataset}_{mode}/model_save"
        plot_folder(folder, xmax=5, dataset=dataset.lower())

    for index in range(1,11):
        plt.subplot(3, 10, index+20)
        folder = f"multi_layer/dnn/{dataset}/{mode}/{index}_10_{dataset}_{mode}/model_save"
        plot_folder(folder, xmax=10, dataset=dataset.lower())

    from pylustrator.helper_functions import axes_to_grid
    axes_to_grid()
    plt.gcf().suptitle(f'{dataset} {mode}', fontsize=16)


plot_rows("CIFAR10", "relu")
plt.savefig(__file__[:-3] + ".pdf")
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.show()
