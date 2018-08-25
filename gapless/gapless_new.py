from sys import path
path.append('../../SurfaceTrapSim/')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from world import World 
from electrode import Electrode, RectElectrode

class CCTWorld(World):
	def __init__(self, axes_permutation=0):
		World.__init__(self, 40, 2*np.pi*18e6)
		if axes_permutation == 0:
			import ad_0 as ad
		elif axes_permutation == 1:
			import ad_1 as ad
		self.derivatives = ad.functions_dict

	def add_electrode(self, name, xr, yr, kind, voltage = 0.0):
		'''
		Add an electrode to the World. Optionally set a voltage on it. Name it with a string.
		kind = 'rf' or 'dc'. If kind == 'rf', then add this electrode to the rf electrode dict
		as well as to the general electrode dict
		'''
		e = Electrode(RectElectrode([xr, yr], self.derivatives))
		World.add_electrode(self, e, name, kind, voltage)

	def drawTrap(self):
		fig, ax = plt.subplots() # initialize figure
		fig.canvas.draw()
		colors = []
		cmap = cm.get_cmap('jet')
		electrode_list = self.dc_electrode_list + self.rf_electrode_list
		elec_colors = cmap(np.linspace(1,0,len(electrode_list)))
		drawLines = []
		for i, nm_e in enumerate(electrode_list):#, unit='electrodes', total=len(self.electrode_dict)):
			# Remember nm_e == (name, elec)
			myElec = nm_e[1].get_baseE()
			# print(nm_e[0],"has %d parts"%len(myElec))
			mycolor = elec_colors[i]
			for subelec in myElec:
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