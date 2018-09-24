from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

__all__ = ['voltProfile', 'plot2D', 'quadru2hess', 'intersectBounds', 'segNeighbor']

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

def quadru2hess(quad):
    hess = np.empty((3,3),'d')
    hess[0,0] = quad[0]-quad[1]
    hess[1,1] = -quad[0]-quad[1]
    hess[2,2] = 2*quad[1]
    hess[0,1] = hess[1,0] = 0.5*quad[2]
    hess[0,2] = hess[2,0] = 0.5*quad[4]
    hess[1,2] = hess[2,1] = 0.5*quad[3]
    return hess

def intersectBounds(bounds):
    lbs = []
    ubs = []
    for bd in bounds:
        if bd is not None:
            lbs.append([bd[l,0] for l in range(3)])
            ubs.append([bd[l,-1] for l in range(3)])
    if len(lbs) > 0:
        lbs = np.max(np.asarray(lbs),0)
        ubs = np.min(np.asarray(ubs),0)
        return np.array([lbs, ubs]).T

def segNeighbor(arr, x, n):
    """
Prequisites:
    arr is an ascending 1D numpy array
    n <= arr.size
    """
    i = np.arange(arr.size)[arr<x][-1]
    if n%2 == 0:
        a = i - (n//2) + 1
        b = i + (n//2) + 1
    else:
        a = i - (n//2)
        b = i + (n//2) + 1
        if x-arr[i] > arr[i+1]-x:
            a = i - (n//2) + 1
            b = i + (n//2) + 2
    if a < 0:
        return np.s_[:b-a]
    if b > arr.size:
        return np.s_[a-b+arr.size:]
    return np.s_[a:b]

def drawTrap(electrode_dict):
    fig, ax = plt.subplots() # initialize figure
    fig.canvas.draw()
    colors = []
    cmap = cm.get_cmap('jet')
    elec_colors = cmap(np.linspace(1,0,len(electrode_dict)))
    drawLines = []
    for i, key in enumerate(electrode_dict):
        myElecs = electrode_dict[key][1].get_baseE()
        mycolor = elec_colors[i]
        # add all subelectrodes to drawLines
        for subelec in myElecs:
            drawLines.append([subelec.x1,subelec.x2])
            drawLines.append([subelec.y2,subelec.y2])
            colors.append(mycolor)
            drawLines.append([subelec.x1,subelec.x2])
            drawLines.append([subelec.y1,subelec.y1])
            colors.append(mycolor)
            drawLines.append([subelec.x1,subelec.x1])
            drawLines.append([subelec.y1,subelec.y2])
            colors.append(mycolor)
            drawLines.append([subelec.x2,subelec.x2])
            drawLines.append([subelec.y1,subelec.y2])
            colors.append(mycolor)
    plt.gca().set_prop_cycle(color=colors)
    drawLines = np.array(drawLines)*1e6
    ax.plot(*drawLines)
    plt.xlabel('x (microns)')
    plt.ylabel('y (microns)')
    plt.show()