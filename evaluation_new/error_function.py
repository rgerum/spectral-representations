import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return x*(x>0)

x = np.arange(10, 30)
y = 20
se = (x-y)**2
plt.subplot(221)
plt.plot(x, se)
plt.semilogx([])

sel = (np.log10(x) - np.log10(y))**2
plt.subplot(222)
plt.plot(x, sel)
plt.semilogx([])

sel = (x/y - 1)**2 + relu(x/y - 1)
plt.subplot(223)
plt.plot(x, sel)
plt.semilogx([])
plt.show()