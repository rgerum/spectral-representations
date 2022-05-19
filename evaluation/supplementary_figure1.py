import numpy as np
import matplotlib.pyplot as plt
from includes.net_helpers import read_data
from includes.color_grad import plot_color_grad
from includes.filter_data import get_corruption_to_level

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Ubuntu Condensed']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Tahoma']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']


def plot(data1, axes):
    for index, (cor, dd) in enumerate(data1.groupby("strength")):
        for i2, (strength, d) in enumerate(dd.groupby("reg_strength")):
            sca(axes[0, index])
            def plot(x, y, **kwargs):
                m = d.groupby(x.name).mean()
                s = d.groupby(x.name).sem()
                x = np.array(m.index)
                yerr = np.array(s[y.name])
                y = np.array(m[y.name])

                plot_color_grad(x, y, x, yerr=yerr, **kwargs)

            if i2 == 0:
                plt.plot(np.mean(d.alpha), np.mean(d.value), "o", label=strength, ms=3, zorder=10)
            else:
                plot(d.alpha, d.value, color1="gray", color2=f"C{i2}", color_label=f"C{i2}", label=strength)

            plt.grid()


def sca(ax):
    ax.figure.sca(ax)


def plot2(data1, axes):
    data1 = data1.query("epoch == 49")
    data1["alpha"] = np.round(data1.alpha, 1)
    data12 = get_corruption_to_level(data1, ["attack_FGSM"])
    plot(data12, axes[0:1])
    for ax, v in zip(axes[0], data12.strength.unique()):
        sca(ax)
        plt.title(f"{v:.2f}"[1:],fontsize=10)
    sca(axes[0, 0])
    plt.ylabel("acc.\nFGSM\nattack")
    data12 = get_corruption_to_level(data1, ["attack_PGD"])
    plot(data12, axes[1:2])
    sca(axes[1, 0])
    plt.ylabel("acc.\nPGD\nattack")
    for ax in axes[1]:
        sca(ax)
        plt.xlabel("$\\alpha$")
        plt.xticks([1,3,5], [1,3,""])
    sca(axes[0, 0])
    plt.text(0, 1.1, "$\\epsilon=$", transform=axes[0, 0].transAxes, ha="right", va="bottom")

if 0:
    fig, axes = plt.subplots(2, 20, sharex=True, sharey=True, figsize=(16/2.54, 4.5/2.54))
    plt.subplots_adjust(bottom=0.25, left=0.14, right=0.98, top=0.79, hspace=0, wspace=0)
    fig.suptitle('shallow MLP $\\cdot$ MNIST', fontsize=10)
    dataset = "mnist"
    #dataset = "cifar10"
    data1 = read_data(r"../../results/expcifar5_DNN_extened_range/iter-{iter:d}_gamma-{gamma}_reg1-{reg1:f}_reg1value-{reg1value:f}/", file_name="data.csv")
    data1 = data1.query("reg_strength > 0.09 or reg_strength < 0.00000001")
    plot2(data1, axes[0:2])

    plt.savefig(__file__[:-3] + "_1.png")
    plt.savefig(__file__[:-3] + "_1.pdf")

fig = plt.figure(0, figsize=(16 / 2.54, 4.5*3 / 2.54))
subfigs = fig.subfigures(3, 1, wspace=0)

fig = subfigs[0]
axes = fig.subplots(2, 20, sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.25, left=0.14, right=0.98, top=0.79, hspace=0, wspace=0)
fig.suptitle('deep MLP $\\cdot$ MNIST', fontsize=10)
data1 = read_data(
    r"../training/logs/deep_mlp/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/",
    file_name="data.csv")
plot2(data1, axes[0:2])
plt.yticks([0, 0.5])

fig = subfigs[1]
axes = fig.subplots(2, 20, sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.25, left=0.14, right=0.98, top=0.79, hspace=0, wspace=0)
fig.suptitle('CNN $\\cdot$ MNIST', fontsize=10)
data1 = read_data(
    r"../training/logs/cnn/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/",
    file_name="data.csv")
plot2(data1, axes[0:2])
plt.yticks([0, 0.5])

fig = subfigs[2]
axes = fig.subplots(2, 20, sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.25, left=0.14, right=0.98, top=0.79, hspace=0, wspace=0)
fig.suptitle('CNN $\\cdot$ CIFAR-10', fontsize=10)
data1 = read_data(
    r"../training/logs/cnn/cifar10/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/",
    file_name="data.csv")
plot2(data1, axes[0:2])
plt.yticks([0, 0.3])

plt.savefig(__file__[:-3] + ".pdf")
plt.show()

