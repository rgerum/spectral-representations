import numpy as np
import matplotlib.pyplot as plt
from includes.net_helpers import read_data
from includes.color_grad import plot_color_grad
from includes.filter_data import get_corruption_to_level

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Ubuntu Condensed']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Tahoma']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']


def do_plot(data1, axes, strengt):
    data1 = data1.query("epoch == 49")
    data1 = get_corruption_to_level(data1, ["attack_FGSM", "attack_PGD"])

    data1["alpha"] = np.round(data1["alpha"], 1)
    data1["strength"] = np.round(data1["strength"], 2)
    data1 = data1.query("reg_strength > 0.09 or reg_strength < 0.00000001")
    index = (data1.strength == 0) & (data1.corrupt == "attack_FGSM")

    data1.loc[index, "corrupt"] = "None"

    data1 = data1.query("strength == "+str(strengt)+" or corrupt == 'None'")

    for index, (cor, dd) in enumerate(data1.groupby("corrupt")):
        if index >= 1:
            plt.sca(axes[1, index-1])
        else:
            plt.sca(axes[0, index])
        for i2, (strength, d) in enumerate(dd.groupby("reg_strength")):

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
            plt.xlabel("alpha")
            plt.ylabel("accuracy")
            plt.grid(True)
            plt.title(cor)
    plt.legend(title="reg_target")
    plt.ylim(0, 1)

# initialize the plot
fig, axes = plt.subplots(6, 2, sharex="row", sharey="row")

# load and plot the deep mlp data
data1 = read_data(r"../training/logs/deep_mlp/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/", file_name="data.csv")
do_plot(data1, axes[:2, :], 0.05)

# load and plot the ccn mnist data
data1 = read_data(r"../training/logs/cnn/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/", file_name="data.csv")
do_plot(data1, axes[2:4, :], 0.10)

# load and plot the ccn cifar10 data
data1 = read_data(r"../training/logs/cnn/cifar10/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/", file_name="data.csv")
do_plot(data1, axes[4:6, :], 0.01)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(13.670000/2.54, 10.560000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[0].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[0].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.131833, 0.712060, 0.213142, 0.234198])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].title.set_fontsize(10)
plt.figure(1).axes[0].title.set_text("no attack")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([-0.323300, 1.063617])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_ha("center")
plt.figure(1).axes[0].texts[1].set_position([-0.521482, 0.204915])
plt.figure(1).axes[0].texts[1].set_rotation(90.0)
plt.figure(1).axes[0].texts[1].set_text("MLP · MNIST")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("validation accuracy")
plt.figure(1).axes[1].set_position([1.013720, 0.605766, 0.346562, 0.233691])
plt.figure(1).axes[2].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[2].set_ylim(0.0, 0.65)
plt.figure(1).axes[2].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[2].set_yticks([0.0, 0.2, 0.4, 0.6])
plt.figure(1).axes[2].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[2].set_yticklabels(["0.0", "0.2", "0.4", "0.6"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[2].set_position([0.529653, 0.712060, 0.213142, 0.234198])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].title.set_fontsize(10)
plt.figure(1).axes[2].title.set_text("FGSM")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([-0.262051, 1.063617])
plt.figure(1).axes[2].texts[0].set_text("b")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("accuracy on attack")
plt.figure(1).axes[3].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[3].set_ylim(0.0, 0.65)
plt.figure(1).axes[3].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[3].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[3].legend(handlelength=1.2999999999999998, handletextpad=0.4, title="strength", fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[3].set_position([0.767662, 0.712060, 0.213142, 0.234198])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].title.set_fontsize(10)
plt.figure(1).axes[3].title.set_text("PGD")
plt.figure(1).axes[3].get_legend()._set_loc((-2.056188, 0.330195))
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_ha("center")
plt.figure(1).axes[3].texts[0].set_position([-0.069909, 1.063617])
plt.figure(1).axes[3].texts[0].set_text("c")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].get_xaxis().get_label().set_text("")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).axes[4].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[4].set_ylim(0.92, 0.985)
plt.figure(1).axes[4].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[4].set_yticks([0.92, 0.94, 0.96, 0.98, 1.0])
plt.figure(1).axes[4].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[4].set_yticklabels(["0.92", "0.94", "0.96", "0.98", "1.00"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[4].set_position([0.131833, 0.405289, 0.213142, 0.234197])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].title.set_text("")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_ha("center")
plt.figure(1).axes[4].texts[0].set_position([-0.323300, 1.067948])
plt.figure(1).axes[4].texts[0].set_text("d")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[1].new
plt.figure(1).axes[4].texts[1].set_ha("center")
plt.figure(1).axes[4].texts[1].set_position([-0.521482, 0.183576])
plt.figure(1).axes[4].texts[1].set_rotation(90.0)
plt.figure(1).axes[4].texts[1].set_text("CNN · MNIST")
plt.figure(1).axes[4].texts[1].set_weight("bold")
plt.figure(1).axes[4].get_xaxis().get_label().set_text("")
plt.figure(1).axes[4].get_yaxis().get_label().set_text("validation accuracy")
plt.figure(1).axes[5].set_position([-1.105865, -0.298368, 0.213142, 0.234197])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].title.set_text("")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([0.270749, 0.381985])
plt.figure(1).axes[5].texts[0].set_text("e")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
plt.figure(1).axes[6].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[6].set_ylim(0.4, 0.75)
plt.figure(1).axes[6].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[6].set_yticks([0.4, 0.5, 0.6, 0.7])
plt.figure(1).axes[6].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[6].set_yticklabels(["0.4", "0.5", "0.6", "0.7"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[6].set_position([0.529653, 0.405289, 0.213142, 0.234197])
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].title.set_text("")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_ha("center")
plt.figure(1).axes[6].texts[0].set_position([-0.262051, 1.067948])
plt.figure(1).axes[6].texts[0].set_text("e")
plt.figure(1).axes[6].texts[0].set_weight("bold")
plt.figure(1).axes[6].get_xaxis().get_label().set_text("")
plt.figure(1).axes[6].get_yaxis().get_label().set_text("accuracy on attack")
plt.figure(1).axes[7].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[7].set_ylim(0.4, 0.75)
plt.figure(1).axes[7].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[7].set_yticks([0.4, 0.5, 0.6, 0.7])
plt.figure(1).axes[7].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[7].set_yticklabels(["0.4", "0.5", "0.6", "0.7"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[7].set_position([0.767662, 0.405289, 0.213142, 0.234197])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[7].title.set_text("")
plt.figure(1).axes[7].get_legend().set_visible(False)
plt.figure(1).axes[7].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[7].transAxes)  # id=plt.figure(1).axes[7].texts[0].new
plt.figure(1).axes[7].texts[0].set_ha("center")
plt.figure(1).axes[7].texts[0].set_position([-0.069909, 1.067948])
plt.figure(1).axes[7].texts[0].set_text("f")
plt.figure(1).axes[7].texts[0].set_weight("bold")
plt.figure(1).axes[7].get_xaxis().get_label().set_text("")
plt.figure(1).axes[7].get_yaxis().get_label().set_text("")
plt.figure(1).axes[8].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[8].set_ylim(0.4, 0.577085)
plt.figure(1).axes[8].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[8].set_yticks([0.4, 0.45, 0.5, 0.55])
plt.figure(1).axes[8].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[8].set_yticklabels(["0.40", "0.45", "0.50", "0.55"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[8].set_position([0.131833, 0.098519, 0.213142, 0.234197])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[8].title.set_text("")
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[8].texts[0].set_ha("center")
plt.figure(1).axes[8].texts[0].set_position([-0.323300, 1.066971])
plt.figure(1).axes[8].texts[0].set_text("g")
plt.figure(1).axes[8].texts[0].set_weight("bold")
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[1].new
plt.figure(1).axes[8].texts[1].set_ha("center")
plt.figure(1).axes[8].texts[1].set_position([-0.521482, 0.097782])
plt.figure(1).axes[8].texts[1].set_rotation(90.0)
plt.figure(1).axes[8].texts[1].set_text("CNN · CIFAR-10")
plt.figure(1).axes[8].texts[1].set_weight("bold")
plt.figure(1).axes[8].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[8].get_yaxis().get_label().set_text("validation accuracy")
plt.figure(1).axes[9].set_position([1.063370, 0.146847, 0.343882, 0.086701])
plt.figure(1).axes[10].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[10].set_ylim(0.0, 0.34)
plt.figure(1).axes[10].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[10].set_yticks([0.0, 0.1, 0.2, 0.3])
plt.figure(1).axes[10].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[10].set_yticklabels(["0.0", "0.1", "0.2", "0.3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[10].set_position([0.529653, 0.098519, 0.213142, 0.234197])
plt.figure(1).axes[10].spines['right'].set_visible(False)
plt.figure(1).axes[10].spines['top'].set_visible(False)
plt.figure(1).axes[10].title.set_text("")
plt.figure(1).axes[10].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[10].transAxes)  # id=plt.figure(1).axes[10].texts[0].new
plt.figure(1).axes[10].texts[0].set_ha("center")
plt.figure(1).axes[10].texts[0].set_position([-0.262051, 1.066971])
plt.figure(1).axes[10].texts[0].set_text("h")
plt.figure(1).axes[10].texts[0].set_weight("bold")
plt.figure(1).axes[10].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[10].get_yaxis().get_label().set_text("accuracy on attack")
plt.figure(1).axes[11].set_xlim(0.42999999999999994, 4.17)
plt.figure(1).axes[11].set_ylim(0.0, 0.34)
plt.figure(1).axes[11].set_xticks([1.0, 2.0, 3.0, 4.0])
plt.figure(1).axes[11].set_yticks([0.0, 0.1, 0.2, 0.3])
plt.figure(1).axes[11].set_xticklabels(["1", "2", "3", "4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[11].set_yticklabels(["0.0", "0.1", "0.2", "0.3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[11].grid(True)
plt.figure(1).axes[11].set_position([0.767662, 0.098519, 0.213142, 0.234197])
plt.figure(1).axes[11].spines['right'].set_visible(False)
plt.figure(1).axes[11].spines['top'].set_visible(False)
plt.figure(1).axes[11].title.set_position([0.436850, 0.099469])
plt.figure(1).axes[11].title.set_text("")
plt.figure(1).axes[11].get_legend().set_visible(False)
plt.figure(1).axes[11].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[11].transAxes)  # id=plt.figure(1).axes[11].texts[0].new
plt.figure(1).axes[11].texts[0].set_ha("center")
plt.figure(1).axes[11].texts[0].set_position([-0.069909, 1.066971])
plt.figure(1).axes[11].texts[0].set_text("i")
plt.figure(1).axes[11].texts[0].set_weight("bold")
plt.figure(1).axes[11].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[11].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".pdf")
plt.show()

