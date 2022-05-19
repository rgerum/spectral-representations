import numpy as np
import matplotlib.pyplot as plt
import pylustrator
from includes.draw import Image

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Ubuntu Condensed']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Tahoma']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']


def linear_fit(x_data, y_data):
    """ calculate the linear regression fit for a list of xy points. """
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    b = np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2)
    a = y_mean - (b * x_mean)
    return a, b

# add a box for the brain path
fig = plt.gcf()
fig.patches.extend([plt.Rectangle((0.25,0.5),0.25,0.25,
                                  fill=True, color='g', alpha=0.5, zorder=0,
                                  transform=fig.transFigure, figure=fig)])

# add the stringer data
plt.axes([0.2,0.2, 0.8,0.8])
data = np.load("additional_data/stringer.npy", allow_pickle=True)[()]
print(data)
ss = data["ss"]
ypred = data["ypred"]
plt.loglog(np.arange(0, ss.size)+1, ss/ss.sum())
plt.loglog(np.arange(0, ss.size)+1, ypred, c='k', lw=0.8)
plt.ylabel("variance")
plt.xlabel("pca dimension")

# add the mouse brain
pylustrator.load("additional_data/mouse_brain.svg")

# add the stringer picture
pylustrator.load("additional_data/stringer_example.png")

# add the mnist picture
pylustrator.load("additional_data/mnist.png")

# add the mpl drawing
draw = Image()
layer1 = draw.addLayer(5, pos=0, color="C0", orientation="vertical")
layer2 = draw.addLayer(3, pos=2, color="C2", orientation="vertical")
layer3 = draw.addLayer(2, pos=4, color="C3", orientation="vertical")

draw.connectLayersDense(layer1, layer2)
draw.connectLayersDense(layer2, layer3)

draw.save()

# plot of the trained spectrum
plt.axes([0.2, 0.2, 0.8, 0.8])
i = 49
y = np.load(f"additional_data/strength-0_epoch-49.npy")
x = np.log(np.arange(1, y.shape[0] + 1, 1.0, y.dtype))
min_x = 5; max_x = 100

max_x_value = 3
a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])

mse = np.mean((b * x[min_x:max_x] + a - y[min_x:max_x]) ** 2)
# return the negative of the slope
l, = plt.loglog(np.exp(x), np.exp(y), "oC1", alpha=0.5, ms=1)
plt.plot(np.exp(x), np.exp(x*b+a), color="gray", lw=0.8)
print("alpha", b)

y = np.load(f"additional_data/strength-1_epoch-49.npy")
x = np.log(np.arange(1, y.shape[0] + 1, 1.0, y.dtype))
min_x = 5; max_x = 100

max_x_value = 3
a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])

mse = np.mean((b * x[min_x:max_x] + a - y[min_x:max_x]) ** 2)
plt.loglog(np.exp(x), np.exp(y), "oC0", alpha=0.5, ms=1)
plt.plot(np.exp(x), np.exp(x*b+a), color="k", lw=0.8)

# plot of the representation matrix
plt.axes([0.2, 0.2, 0.8, 0.8])
np.random.seed(1234)
im = np.random.rand(3, 5)
plt.imshow(im)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.930000/2.54, 8.850000/2.54, forward=True)
plt.figure(1).ax_dict["additional_data/mnist.png"].set_position([0.025994, 0.091673, 0.171763, 0.251279])
plt.figure(1).ax_dict["additional_data/mnist.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["additional_data/mnist.png"].transAxes)  # id=plt.figure(1).ax_dict["additional_data/mnist.png"].texts[0].new
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[0].set_position([-0.100001, 1.057610])
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[0].set_text("e")
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["additional_data/mnist.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["additional_data/mnist.png"].transAxes)  # id=plt.figure(1).ax_dict["additional_data/mnist.png"].texts[1].new
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[1].set_position([-0.100001, 1.315250])
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[1].set_text("computer vision")
plt.figure(1).ax_dict["additional_data/mnist.png"].texts[1].set_weight("bold")
plt.figure(1).ax_dict["draw_mlp_2.png"].set_position([0.255925, 0.041147, 0.294271, 0.430413])
plt.figure(1).ax_dict["draw_mlp_2.png"].set_zorder(-1)
plt.figure(1).ax_dict["draw_mlp_2.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["draw_mlp_2.png"].transAxes)  # id=plt.figure(1).ax_dict["draw_mlp_2.png"].texts[0].new
plt.figure(1).ax_dict["draw_mlp_2.png"].texts[0].set_position([-0.851409, 2.179046])
plt.figure(1).ax_dict["draw_mlp_2.png"].texts[0].set_text("f")
plt.figure(1).ax_dict["draw_mlp_2.png"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["draw_mlp_2.png"].texts[10].set_visible(False)
plt.figure(1).ax_dict["mouse_brain.svg"].set_position([0.242173, 0.498354, 0.291006, 0.490280])
plt.figure(1).ax_dict["mouse_brain.svg"].set_zorder(1)
plt.figure(1).ax_dict["mouse_brain.svg"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["mouse_brain.svg"].transAxes)  # id=plt.figure(1).ax_dict["mouse_brain.svg"].texts[0].new
plt.figure(1).ax_dict["mouse_brain.svg"].texts[0].set_position([0.375480, 0.942689])
plt.figure(1).ax_dict["mouse_brain.svg"].texts[0].set_text("V1")
plt.figure(1).ax_dict["mouse_brain.svg"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["mouse_brain.svg"].transAxes)  # id=plt.figure(1).ax_dict["mouse_brain.svg"].texts[1].new
plt.figure(1).ax_dict["mouse_brain.svg"].texts[1].set_position([-0.055951, 0.942689])
plt.figure(1).ax_dict["mouse_brain.svg"].texts[1].set_text("b")
plt.figure(1).ax_dict["mouse_brain.svg"].texts[1].set_weight("bold")
plt.figure(1).ax_dict["additional_data/stringer_example.png"].set_position([0.025994, 0.565817, 0.171763, 0.251279])
plt.figure(1).ax_dict["additional_data/stringer_example.png"].set_zorder(1)
plt.figure(1).ax_dict["additional_data/stringer_example.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["additional_data/stringer_example.png"].transAxes)  # id=plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[0].new
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[0].set_position([-0.100001, 1.120001])
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[0].set_text("a")
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["additional_data/stringer_example.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["additional_data/stringer_example.png"].transAxes)  # id=plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[1].new
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[1].set_position([-0.100001, 1.531752])
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[1].set_text("biological vision")
plt.figure(1).ax_dict["additional_data/stringer_example.png"].texts[1].set_weight("bold")
plt.figure(1).axes[0].set_xlim(0.7071067811865475, 1448.1546878700499)
plt.figure(1).axes[0].set_ylim(1.002515179763185e-05, 0.7)
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.706632, 0.659973, 0.259255, 0.297458])
plt.figure(1).axes[0].set_zorder(1)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.373791, 1.010434])
plt.figure(1).axes[0].texts[0].set_text("c")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([0.073069, 0.872324])
plt.figure(1).axes[0].texts[1].set_text("$\\alpha = 1$")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("index")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("variance")
plt.figure(1).axes[5].set_xlim(0.7071067811865475, 1448.1546878700499)
plt.figure(1).axes[5].set_ylim(1.002515179763185e-05, 0.7)
plt.figure(1).axes[5].grid(True)
plt.figure(1).axes[5].set_position([0.706632, 0.142551, 0.259255, 0.297458])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.373791, 0.935394])
plt.figure(1).axes[5].texts[0].set_text("e")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[1].new
plt.figure(1).axes[5].texts[1].set_position([0.028597, 0.516485])
plt.figure(1).axes[5].texts[1].set_text("$\\alpha = 1$")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[2].new
plt.figure(1).axes[5].texts[2].set_color("#888a85ff")
plt.figure(1).axes[5].texts[2].set_position([0.375478, 0.694316])
plt.figure(1).axes[5].texts[2].set_text("$\\alpha = 1.6$")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("index")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("variance")
plt.figure(1).axes[6].set_xlim(-0.5, 4.5)
plt.figure(1).axes[6].set_ylim(2.5, -0.5)
plt.figure(1).axes[6].set_xticks([])
plt.figure(1).axes[6].set_yticks([])
plt.figure(1).axes[6].set_xticklabels([], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[6].set_yticklabels([], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[6].set_position([0.436139, 0.783401, 0.133538, 0.117178])
plt.figure(1).axes[6].set_zorder(1)
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_ha("center")
plt.figure(1).axes[6].texts[0].set_position([-0.109617, 0.749215])
plt.figure(1).axes[6].texts[0].set_text("1")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[1].new
plt.figure(1).axes[6].texts[1].set_ha("center")
plt.figure(1).axes[6].texts[1].set_position([-0.149875, 0.471244])
plt.figure(1).axes[6].texts[1].set_rotation(90.0)
plt.figure(1).axes[6].texts[1].set_text("...")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[2].new
plt.figure(1).axes[6].texts[2].set_ha("center")
plt.figure(1).axes[6].texts[2].set_position([-0.109617, 0.068667])
plt.figure(1).axes[6].texts[2].set_text("$N$")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[3].new
plt.figure(1).axes[6].texts[3].set_position([0.039912, 1.161377])
plt.figure(1).axes[6].texts[3].set_text("1")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[4].new
plt.figure(1).axes[6].texts[4].set_position([0.229698, 1.161377])
plt.figure(1).axes[6].texts[4].set_text("...")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[5].new
plt.figure(1).axes[6].texts[5].set_position([0.845066, 1.161377])
plt.figure(1).axes[6].texts[5].set_text("$f$")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[6].new
plt.figure(1).axes[6].texts[6].set_ha("center")
plt.figure(1).axes[6].texts[6].set_position([0.482747, 1.516029])
plt.figure(1).axes[6].texts[6].set_text("neurons")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[7].new
plt.figure(1).axes[6].texts[7].set_ha("center")
plt.figure(1).axes[6].texts[7].set_position([-0.355956, 0.096617])
plt.figure(1).axes[6].texts[7].set_rotation(90.0)
plt.figure(1).axes[6].texts[7].set_text("images")
plt.figure(1).patches[0].set_edgecolor("#babdb6ff")
plt.figure(1).patches[0].set_facecolor("#babdb6ff")
plt.figure(1).patches[0].set_height(0.536121)
plt.figure(1).patches[0].set_width(1.018173)
plt.figure(1).patches[0].set_xy([-0.009776, 0.479452])
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".pdf")
plt.show()
