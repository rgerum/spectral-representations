import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from includes.net_helpers import read_data
from includes.color_grad import plot_color_grad
from includes.lab_colormap import LabColormap
from includes.filter_data import get_corruption_to_level

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Ubuntu Condensed']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Tahoma']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']


data1 = read_data(r"../training/logs/shallow_mlp/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/", file_name="data.csv")

data0 = data1
data1 = data1.query("epoch == 49")
data1 = get_corruption_to_level(data1, ["attack_FGSM", "attack_PGD"])

parent_fig, axs = plt.subplots(2, 1, figsize=(14/2.54, (8.95+4.5)/2.54), gridspec_kw={'height_ratios': [8.95, 4.5]})
gridspec = axs[0].get_subplotspec().get_gridspec()

# clear the left column for the subfigure:
for a in axs:
    a.remove()

fig = parent_fig.add_subfigure(gridspec[0])

axes = fig.subplots(3, 2, sharex="row", sharey="row")

data1["alpha"] = np.round(data1["alpha"], 1)
data1 = data1.query("reg_strength > 0.09 or reg_strength < 0.00000001")
index = (data1.strength == 0) & (data1.corrupt == "attack_FGSM")

data1.loc[index, "strength"] = 0.05
data1.loc[index, "corrupt"] = "None"

data1 = data1.query("strength == 0.05")


def sca(ax):
    ax.figure.sca(ax)

for index, (cor, dd) in enumerate(data1.groupby("corrupt")):
    if index >= 1:
        sca(axes[1, index-1])
    else:
        sca(axes[0, index])
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


for index, (cor, dd) in enumerate(data0.query("reg_strength > 0.09").groupby("reg_strength")):
    def plot(x, y, **kwargs):
        m = d.groupby(x.name).mean()
        s = d.groupby(x.name).sem()
        c, = plt.plot(m.index, m[y.name], **kwargs)
        plt.fill_between(m.index, m[y.name]-s[y.name], m[y.name]+s[y.name], alpha=0.5, color=c.get_color())

    sca(axes[2, index])
    N = len(dd.groupby("reg_target"))
    if index == 0:
        cmap = LabColormap(["gray", "C1"], N)
    else:
        cmap = LabColormap(["gray", "C2"], N)
    cmap = matplotlib.cm.get_cmap(cmap, N)
    c = []
    for i2, (strength, d) in enumerate(dd.groupby("reg_target")):
        plot(d.epoch, d["alpha"], label="acc", color=cmap(i2))
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("alpha")
        c.append(strength)
    plt.title(cor)
    dummie_cax = plt.scatter(c, c, c=c, s=0, cmap=cmap)
    c = fig.colorbar(dummie_cax, ticks=[1,2,3,4,5], label=f"colorbar{index}")
    c.ax.set_label(f"colorbar{index}")
    if index != 0:
        c.set_label('target alpha')

### lower part

fig2 = parent_fig.add_subfigure(gridspec[1])


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

def plot2(data1, axes):
    data1 = data1.query("epoch == 49")
    data1["alpha"] = np.round(data1.alpha, 1)
    data12 = get_corruption_to_level(data1, ["attack_FGSM"])

    data12 = data12.query("strength <= 0.10")

    plot(data12, axes[0:1])
    for ax, v in zip(axes[0], data12.strength.unique()):
        sca(ax)
        plt.title(f"{v:.2f}",fontsize=10)
    sca(axes[0, 0])
    plt.ylabel("acc. on\nFGSM\nattack")
    data12 = get_corruption_to_level(data1, ["attack_PGD"])
    data12 = data12.query("strength <= 0.10")

    plot(data12, axes[1:2])
    sca(axes[1, 0])
    plt.ylabel("acc. on\nPGD\nattack")
    for iii, ax in enumerate(axes[1]):
        sca(ax)
        if iii == 5:
            plt.xlabel("measured $\\alpha$")
        plt.xticks([1,3,5], [1,3,5])
    sca(axes[0, 0])
    plt.text(0, 1.12, "$\\epsilon=$", transform=axes[0, 0].transAxes, ha="right", va="bottom")
    plt.text(-1., 1.2, "f", transform=axes[0, 0].transAxes, ha="right", va="bottom").set_weight("bold")

    for ax in axes.ravel():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

axes = fig2.subplots(2, 11, sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.25, left=0.15, right=0.99, top=0.83, hspace=0.08, wspace=0.08)

data1 = read_data(
    r"../training/logs/shallow_mlp/mnist/repeat-{repeat:d}_reg-strength-{reg_strength:f}_reg-target-{reg_target:f}/",
    file_name="data.csv")
data1 = data1.query("reg_strength > 0.09 or reg_strength < 0.00000001")
plot2(data1, axes[0:2])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
#plt.figure(1).set_size_inches(13.990000/2.54, 8.950000/2.54, forward=True)
plt.figure(1).ax_dict["colorbar0"].set_ylim(0.6, 5.0)
plt.figure(1).ax_dict["colorbar0"].set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).ax_dict["colorbar0"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).ax_dict["colorbar0"].set_position([0.890092, 0.162728, 0.008120, 0.253992])
plt.figure(1).ax_dict["colorbar0"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["colorbar1"].set_ylim(0.6, 5.0)
plt.figure(1).ax_dict["colorbar1"].set_yticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).ax_dict["colorbar1"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).ax_dict["colorbar1"].set_position([0.914666, 0.162728, 0.008120, 0.253992])
plt.figure(1).ax_dict["colorbar1"].get_yaxis().get_label().set_text("$\\alpha_\mathrm{target}$")
plt.figure(1).axes[0].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[0].set_ylim(0.9262192963248304, 0.945594777178565)
plt.figure(1).axes[0].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[0].set_yticks([0.93, 0.935, 0.94, 0.945])
plt.figure(1).axes[0].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[0].set_yticklabels(["0.930", "0.935", "0.940", "0.945"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[0].set_position([0.112188, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[0].set_yticklabels(["", ""], minor=True)
plt.figure(1).axes[0].set_yticks([0.935, 0.945], minor=True)
plt.figure(1).axes[0].set_zorder(0)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].title.set_fontsize(10)
plt.figure(1).axes[0].title.set_text("no attack")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.449291, 1.063617])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("validation accuracy")
plt.figure(1).axes[1].set_position([1.023559, 0.499289, 0.355019, 0.296493])
plt.figure(1).axes[2].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[2].set_ylim(0.0, 1.0)
plt.figure(1).axes[2].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[2].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[2].set_position([0.519714, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[2].set_zorder(1)
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].title.set_fontsize(10)
plt.figure(1).axes[2].title.set_text("FGSM")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.352252, 1.063617])
plt.figure(1).axes[2].texts[0].set_text("b")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[1].new
plt.figure(1).axes[2].texts[1].set_fontsize(8)
plt.figure(1).axes[2].texts[1].set_ha("center")
plt.figure(1).axes[2].texts[1].set_position([-0.757648, 0.894407])
plt.figure(1).axes[2].texts[1].set_text("baseline")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[2].new
plt.figure(1).axes[2].texts[2].set_fontsize(8)
plt.figure(1).axes[2].texts[2].set_ha("center")
plt.figure(1).axes[2].texts[2].set_position([-0.757648, 0.676074])
plt.figure(1).axes[2].texts[2].set_style("italic")
plt.figure(1).axes[2].texts[2].set_text("weak")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[3].new
plt.figure(1).axes[2].texts[3].set_fontsize(8)
plt.figure(1).axes[2].texts[3].set_ha("center")
plt.figure(1).axes[2].texts[3].set_position([-0.757648, 0.454221])
plt.figure(1).axes[2].texts[3].set_text("strong")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("accuracy on attack")
plt.figure(1).axes[3].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[3].set_ylim(0.0, 1.0)
plt.figure(1).axes[3].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[3].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[3].legend(labelspacing=0.9, handlelength=1.2, handletextpad=0.4, title="strength", fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[3].set_position([0.763531, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].title.set_fontsize(10)
plt.figure(1).axes[3].title.set_text("PGD")
plt.figure(1).axes[3].get_legend()._set_loc((-2.117189, 0.389776))
plt.figure(1).axes[3].get_legend()._set_loc((-2.105202, 0.277088))
plt.figure(1).axes[3].get_legend()._set_loc((-2.084161, 0.312303))
plt.figure(1).axes[3].get_legend()._set_loc((-2.108709, 0.280610))
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.141299, 1.063617])
plt.figure(1).axes[3].texts[0].set_text("c")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].get_xaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).axes[4].set_ylim(0.0, 5.222513057554038)
plt.figure(1).axes[4].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[4].set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[4].set_position([0.112188, 0.133400, 0.355019, 0.296493])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].title.set_text("")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_position([-0.181456, 1.011562])
plt.figure(1).axes[4].texts[0].set_text("d")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[1].new
plt.figure(1).axes[4].texts[1].set_fontsize(8)
plt.figure(1).axes[4].texts[1].set_position([0.064331, 0.826530])
plt.figure(1).axes[4].texts[1].set_text("weak")
plt.figure(1).axes[4].get_yaxis().get_label().set_text("measured $\\alpha$")
plt.figure(1).axes[5].set_ylim(0.0, 5.222513057554038)
plt.figure(1).axes[5].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[5].set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[5].set_position([0.519714, 0.133400, 0.355019, 0.296493])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].title.set_text("")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.070856, 1.011562])
plt.figure(1).axes[5].texts[0].set_text("e")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[1].new
plt.figure(1).axes[5].texts[1].set_fontsize(8)
plt.figure(1).axes[5].texts[1].set_position([0.064331, 0.826530])
plt.figure(1).axes[5].texts[1].set_text("strong")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".pdf")
plt.show()

