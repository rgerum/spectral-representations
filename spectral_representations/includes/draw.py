import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
import numpy as np


def darken(color, f, a=None):
    from matplotlib import colors
    c = np.array(colors.to_rgba(color))
    if f > 0:
        c2 = np.zeros(3)
    else:
        c2 = np.ones(3)
    c[:3] = c[:3] * (1 - np.abs(f)) + np.abs(f) * c2
    if a is not None:
        c[3] = a
    return c

@dataclass
class Object:
    x: float = 0
    y: float = 0
    name: str = ""
    artist: str = ""

Node_Radius = 0.35
rotated = False

class Image:
    def __init__(self):
        #self.fig = plt.figure()
        self.ax = plt.axes([0, 0, 1, 1], label="draw_mlp_2.png")

    def ensureSpace(self, rect, radius=None):
        plt.sca(self.ax)
        if len(rect) == 2:
            xmin, ymin = rect
            xmax, ymax = rect
            if radius is not None:
                if isinstance(radius, (int, float)):
                    xmin -= radius
                    xmax += radius
                    ymin -= radius
                    ymax += radius
                else:
                    xmin -= radius[0]
                    xmax += radius[0]
                    ymin -= radius[1]
                    ymax += radius[1]
        else:
            xmin, ymin, xmax, ymax = rect
        xlim = plt.gca().get_xlim()
        plt.xlim(min(xlim[0], xmin), max(xlim[1], xmax))
        ylim = plt.gca().get_ylim()
        plt.ylim(min(ylim[0], ymin), max(ylim[1], ymax))


    def plotBox(self, x, y, name, color="C0", height=1, width=1, text2=None):
        plt.sca(self.ax)
        if rotated is True:
            obj = Object(y, x, name)
        else:
            obj = Object(x, y, name)
        ax = plt.gca()
        ax.set_aspect(1)

        obj.artist = plt.Rectangle((obj.x - width/2, obj.y - height/2), height=height, width=width, facecolor=darken(color, -0.2),
                                edgecolor=darken(color, 0.6), lw=2)
        ax.add_artist(obj.artist)
        plt.text(obj.x, obj.y + 0.10, name, va="center", ha="center").data_fontsize = 0.17
        if text2 is not None:
            plt.text(obj.x, obj.y - 0.15, text2, va="center", ha="center").data_fontsize = 0.17

        self.ensureSpace([obj.x, obj.y], [width, height])

        return obj

    def plotNodes(self, x, y, name, color="C0"):
        plt.sca(self.ax)
        if rotated is True:
            obj = Object(y, x, name)
        else:
            obj = Object(x, y, name)
        ax = plt.gca()
        ax.set_aspect(1)
        if 1:
            obj.artist = plt.Circle((obj.x, obj.y), Node_Radius, facecolor=darken(color, -0.2), edgecolor=darken(color, 0.6), lw=1)
            ax.add_artist(obj.artist)
        else:
            obj.artist = plt.Rectangle((obj.x - Node_Radius, obj.y - Node_Radius), height=Node_Radius * 2, width=Node_Radius*2, facecolor=darken(color, -0.2),
                                    edgecolor=darken(color, 0.6), lw=2)
            ax.add_artist(obj.artist)
        plt.text(obj.x, obj.y, name, va="center", ha="center")

        self.ensureSpace([x, y], Node_Radius*2)

        return obj

    def plotArrow(self, obj1, obj2, color=None, zorder=1):
        plt.sca(self.ax)
        def movePointOut(obj, dir):
            p = np.array([obj.x, obj.y])
            if isinstance(obj.artist, mpl.patches.Circle):
                return p + dir * obj.artist.radius
            if isinstance(obj.artist, mpl.patches.Rectangle):
                lx = obj.artist.get_width() / 2 / dir[0]
                ly = obj.artist.get_height() / 2 / dir[1]
                if abs(lx) < abs(ly):
                    return p + dir * abs(lx)
                return p + dir * abs(ly)
            return p
        p1 = np.array([obj1.x, obj1.y], dtype=float)
        p2 = np.array([obj2.x, obj2.y], dtype=float)
        d = p2 - p1
        length = np.linalg.norm(d)
        d /= length
        offset1 = 0.08
        offset2 = offset1
        pos_start = movePointOut(obj1, d) + offset1 * d
        pos_end = movePointOut(obj2, -d) - offset2 * d
        delta = pos_end - pos_start
        plt.arrow(pos_start[0], pos_start[1], delta[0], delta[1], head_width=0.1, head_length=0.1, fc=color, length_includes_head=True, color=color, zorder=zorder, lw=0.8)

    def addLayer(self, N, pos, text="", color="C0", orientation="horizontal"):
        plt.sca(self.ax)
        if not callable(color):
            color = lambda x, c=color: c
        if not callable(text):
            text = lambda x, c=text: c
        if orientation == "horizontal":
            return [self.plotNodes(i - N / 2 + 0.5, pos, text(i), color(i)) for i in range(N)]
        else:
            return [self.plotNodes(pos, i - N / 2 + 0.5, text(i), color(i)) for i in range(N)]

    def addText(self, x, y, text):
        plt.sca(self.ax)
        plt.text(x, y, text, va="center", ha="center")
        self.ensureSpace([x-0.5, y-0.5, x+0.5, y+0.5])

    def connectLayersDense(self, layer1, layer2, color="k", zorder=1):
        plt.sca(self.ax)
        if not callable(color):
            color = lambda x, y, c=color: c
        if not callable(zorder):
            zorder = lambda x, y, c=zorder: c
        for i in range(len(layer1)):
            for j in range(len(layer2)):
                self.plotArrow(layer1[i], layer2[j], color=color(i, j), zorder=zorder(i, j))

    def connectLayersConv(self, layer1, layer2, kernel, color="k", zorder=1):
        plt.sca(self.ax)
        if not callable(color):
            color = lambda x, y, c=color: c
        if not callable(zorder):
            zorder = lambda x, y, c=zorder: c
        for i in range(len(layer1)):
            for j in range(len(layer2)):
                if abs(i-j) < kernel/2:
                    self.plotArrow(layer1[i], layer2[j], color=color(i, j), zorder=zorder(i, j))

    def crop_figure(self):
        plt.sca(self.ax)
        # crop the figure to only contain the axes
        size = plt.gcf().get_size_inches()
        ((x1, y1), (x2, y2)) = np.array(plt.gca().get_position())
        w = x2-x1
        h = y2-y1
        plt.gcf().set_size_inches(size[0]*w, size[1]*h)
        plt.gca().set_position([0, 0, 1, 1])
        plt.gca().set_axis_off()

        # scale the figure to keep the datascale constant
        size = plt.gcf().get_size_inches()
        ymin, ymax = plt.gca().get_ylim()
        yscale = (ymax - ymin) / size[1] * 0.65
        plt.gcf().set_size_inches(size[0]*yscale, size[1]*yscale)

    def save(self):
        plt.sca(self.ax)
        self.crop_figure()
        #plt.savefig(filename, dpi=dpi)

    def show(self):
        plt.sca(self.ax)
        plt.show()
