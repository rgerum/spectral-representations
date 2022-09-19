import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_color_grad(x, y, c, color1=None, color2=None, color_label=None, yerr=None, label=None, N=10, **kwargs):
    line1, = plt.plot([], [], color=color1 if color_label is None else color_label, label=label)
    if color_label is None:
        color1 = np.asarray(mpl.colors.to_rgba(line1.get_color()))
    else:
        color1 = np.asarray(mpl.colors.to_rgba(color1))
    if color2 is None:
        color2 = np.asarray(mpl.colors.to_rgba("w"))*0.9 + color1*0.1
    else:
        color2 = np.asarray(mpl.colors.to_rgba(color2))
    indices = np.argsort(c)
    if len(indices) < N:
        N = len(indices)
    for i in range(N):
        i1, i2 = int(len(indices)/N*i), int(len(indices)/N*(i+1))
        ind = indices[i1:i2+1]
        f = i/(N-1)
        color = color1*(1-f) + color2*f
        plt.plot(x[ind], y[ind], color=color, **kwargs)
        if yerr is not None:
            plt.fill_between(x[ind], y[ind]-yerr[ind], y[ind]+yerr[ind], color=color, alpha=0.5, edgecolor="none")

