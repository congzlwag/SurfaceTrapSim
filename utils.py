from matplotlib import pyplot as plt
import numpy as np

__all__ = ['voltProfile', '']

def voltProfileSinglet(voltage, title, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    volt = np.log10(abs(voltage[voltage!=0]))
    ax.hist(volt)
    ax.set_title(title)
    ax.set_xlabel("lg(|V|)")

def voltProfile(voltages, titles):
    if voltages.ndim == 1:
        if isinstance(titles, list):
            title = titles[0]
        voltProfileSinglet(voltages, title)
        return
    elif voltages.ndim > 2:
        voltages.shape = (voltages.shape[0], -1)

    nn = voltages.shape[0]
    assert len(titles) == nn
    na = np.ceil(nn**0.5)
    nb = np.ceil(nn / float(na))
    fig, axes = plt.subplots(na, nb)
    axes = axes.ravel()
    for i in range(nn):
        voltProfileSinglet(voltages[i], titles[i], axes[i])
    plt.show()

def plot2D(func, xr, yr, ax=None):
    data = np.asarray([[func(x,y) for y in yr] for x in xr])
    data.shape = data.shape[:2]
    if ax is None:
        ax = plt.subplot(111)
    im = ax.imshow(data.T, origin='lower', cmap='bone_r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(np.arange(xr.size)[::10])
    ax.set_xticklabels(["%.3f"%x for x in xr[::10]])
    ax.set_yticks(np.arange(yr.size)[::10])
    ax.set_yticklabels(["%.3f"%x for x in yr[::10]])
    plt.colorbar(im)
    return ax # You can customize the axis afterwards as you wish