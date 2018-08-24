from __future__ import print_function
from scipy.interpolate import Akima1DInterpolator, RegularGridInterpolator
from scipy.linalg import norm
import numpy as np
import sys
# import pickle
# import numdifftools as nd
# import tqdm

class Electrode:
	"""
	General Electrode, packing a basic electrode instance in order to
		a. designate voltage
		b. expand multipoles
		c. unite different basic electrodes that share one terminal
	
	Attributes:
		e: root basic electrode. Can be any basic electrode instance, as long as:
			hasattr(e,'pot') and hasattr(e,'grad') and hasattr(e,'hessian')
		volt: voltage on this (set of) electrode(s)
		_expand_pos: coordinates of the expanding position
		taylor_dict: coefficients in the taylor expansion at expand_pos
	
	Conventions:
		Axes convention
			x: perpendicular to the trap axis and parallel to the surface trap
			y: perpendicular to the surface trap
			z: axial
		Multipole convention
			Da: Da An's convention, used in cct
			Littich: Convention in Gebhard Littich thesis, p.39

	"""
	multipole_convention = 'Da'
	def __init__(self, electrode, volt=0):
		self._e = electrode
		self.volt = volt
		self._sub_electrodes = []

	def pot(self, r):
		res = self._e.pot(r)
		for elec in self._sub_electrodes:
			res += elec.pot(r)
		return res

	def grad(self, r):
		res = self._e.grad(r)
		for elec in self._sub_electrodes:
			res += elec.grad(r)
		return res

	def hessian(self,r):
		res = self._e.hessian(r)
		for elec in self._sub_electrodes:
			res += elec.hessian(r)
		return res

	def compute_potential(self, r):
		return self.volt*self.pot(r)

	def compute_electric_field(self, r):
		'''
		Calculate the electric field at the observation point, given the voltage on the electrode
		If voltage is set in Volts, field is in Volts/meter.
		E = -grad(Potential)
		'''
		return -(self.volt)*self.grad(r)

	def compute_hessian(self, r):
		return self.volt*self.hessian(r)

	def compute_d_effective(self, r):
		'''
		Calculate the effective distance due to this electrode. This is defined
		as the parallel plate capacitor separation which gives the observed electric
		field for the given applied voltage. That is,
		Deff = V/E. Will be different in each direction so we return [deff_x, deff_y, deff_z]
		'''
		return 1/norm(self.grad(r)) # in meters

	def expand_potential(self, r):
		"""
		Numerically expand the potential due to the electrode to second order as a taylor series
		around the obersvation point r = [x, y, z]

		self.taylor_dict is a dictionary containing the terms of the expansion. e.g.
		self.taylor_dict['x^2'] = (1/2)d^2\phi/dx^2
		"""
		if hasattr(self,"taylor_dict") and hasattr(self, '_expand_pos') and np.allclose(self._expand_pos,r,1e-10):
			return

		# Otherwise we need to gather the taylor_dict for this NEW position
		# Clear out multipole_dict
		self.multipole_dict = {}
		# Gather
		self._expand_pos = r
		self.taylor_dict = {}
		self.taylor_dict['C'] = self.pot(r)
		self.taylor_dict['x'], self.taylor_dict['y'], self.taylor_dict['z'] = self.grad(r)
		# second derivatives
		hessian = self.hessian(r)
		if Electrode.multipole_convention=="Da":
			self.taylor_dict['x^2'] = 0.25*hessian[0,0]
			self.taylor_dict['y^2'] = 0.25*hessian[1,1]
			self.taylor_dict['z^2'] = 0.25*hessian[2,2]
			self.taylor_dict['xy'] = 0.25*hessian[0,1]
			self.taylor_dict['xz'] = 0.25*hessian[0,2]
			self.taylor_dict['zy'] = 0.25*hessian[1,2]
		elif Electrode.multipole_convention=="Littich":
			self.taylor_dict['x^2'] = 0.5*hessian[0,0]
			self.taylor_dict['y^2'] = 0.5*hessian[1,1]
			self.taylor_dict['z^2'] = 0.5*hessian[2,2]
			self.taylor_dict['xy'] = hessian[0,1]
			self.taylor_dict['xz'] = hessian[0,2]
			self.taylor_dict['zy'] = hessian[1,2]

		# # higher order stuff
		# self.taylor_dict['z^3'], self.taylor_dict['xz^2'], self.taylor_dict['yz^2'] = self.third_order_derivatives(r)
		# self.taylor_dict['z^4'], self.taylor_dict['x^2z^2'], self.taylor_dict['y^2z^2'] = self.fourth_order_derivatives(r)

	def get_multipole(self, multipole, r0=1):
		if Electrode.multipole_convention=='Da':
			if multipole=='U1':
				return 0.5*(r0**2)*(self.taylor_dict['x^2'] - self.taylor_dict['y^2'])
			if multipole=='U2':
				return 0.5*(r0**2)*(2 * self.taylor_dict['z^2'] - self.taylor_dict['x^2'] - self.taylor_dict['y^2'])
			if multipole=='U3':
				return (r0**2)*self.taylor_dict['xy']
			if multipole=='U4':
				return (r0**2)*self.taylor_dict['zy']
			if multipole=='U5':
				return (r0**2)*self.taylor_dict['xz']
		elif Electrode.multipole_convention=='Littich':
			if multipole=='U1':
				return (r0**2)*(self.taylor_dict['x^2'] - self.taylor_dict['y^2'])
			if multipole=='U2':
				return (r0**2)*(self.taylor_dict['z^2'])
			if multipole=='U3':
				return 2*(r0**2)*self.taylor_dict['xy']
			if multipole=='U4':
				return 2*(r0**2)*self.taylor_dict['zy']
			if multipole=='U5':
				return 2*(r0**2)*self.taylor_dict['xz']

		# fields
		if multipole=='Ex':
			return -1*r0*self.taylor_dict['x']
		if multipole=='Ey':
			return -1*r0*self.taylor_dict['y']
		if multipole=='Ez':
			return -1*r0*self.taylor_dict['z']

		if multipole=='C':
			return self.taylor_dict['C']

	def expand_in_multipoles(self, r, controlled_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], r0=1):
		"""
		Obtain the multipole expansion for the potential due to the electrode at the observation point.
		Note that U1,U2 have a 2x functional form compared to U3,U4,U5 to equalize curvature
		"""
		# first, make sure taylor expansion at position r is done
		self.expand_potential(r)

		for cm in controlled_multipoles:
			if not cm in self.multipole_dict.keys():
				self.multipole_dict[cm] = self.get_multipole(cm, r0)
		
		# # stuff that isn't really multipoles

		# self.multipole_dict['z^2'] = (r0)**2*2*self.taylor_dict['z^2']

		# self.multipole_dict['z^4'] = (r0)**4*self.taylor_dict['z^4']/24.
		# self.multipole_dict['xz^2'] = (r0)**3 * self.taylor_dict['xz^2']
		# self.multipole_dict['yz^2'] = (r0)**3 * self.taylor_dict['yz^2']

		# #These terms give corrections to (radial frequency)**2 along the z axis:
		# self.multipole_dict['x^2z^2']  =  (r0)**4 * self.taylor_dict['x^2z^2']
		# self.multipole_dict['y^2z^2']  =  (r0)**4 * self.taylor_dict['y^2z^2']

	def get_expand_pos(self):
		return self._expand_pos

	def extend(self, new_elec):
		self._sub_electrodes.append(new_elec)

	def get_region_bounds(self):
		lbs = []
		ubs = []
		for elec in [self._e]+self._sub_electrodes:
			if hasattr(elec, "gvec"):
				lbs.append([elec.gvec[l][:2].mean() for l in range(3)])
				ubs.append([elec.gvec[l][-2:].mean() for l in range(3)])
		if len(lbs) > 0:
			lbs = np.max(np.asarray(lbs),0)
			ubs = np.min(np.asarray(ubs),0)
			return lbs, ubs

	def output_hessdata(self):
		return self._e.output_hessdata()

class RRPESElectrode:
	"""
	PESElectrode: Rectangular Region Poisson Equation Solved Electrode
	
	Attributes:
		gvec: grid vectors
		_data:
		_grad_data:
		_hess_data:
		_interpolant:
		_grad_interpolants:
		_hess_interpolants:

	For future Developer(s): If one day either 3D Akima or 3D spline is available in scipy.interpolate
		a. Replace the RegularGridInterpolator method here
		b. Translate the functions in interp_diff.m to this class
	"""
	def __init__(self, gvec, pot_data, grad_data=None, hess_data=None):
		self.gvec = gvec
		self._data = pot_data
		if grad_data is not None:
			self._grad_data = grad_data.reshape((3,)+pot_data.shape)
		if hess_data is not None:
			self._hess_data = {}
			axes_nm = ['x','y','z']
			for i in range(3):
				ai = axes_nm[i]
				self._hess_data[(i,i)] = hess_data[ai+ai].reshape(pot_data.shape)
				for j in range(i+1,3):
					try:
						self._hess_data[(i,j)] = hess_data[ai+axes_nm[j]].reshape(pot_data.shape)
					except:
						self._hess_data[(i,j)] = hess_data[axes_nm[j]+ai].reshape(pot_data.shape)

	def finer_grid(self, strides):
		"""
		To be deprecated due to low speed compared to matlab
		Future dev. with 3D spline implemented in scipy.interpolate should intergrate the functions in interp_diff.m here
		"""
		strides = np.asarray(strides)
		incres = np.asarray([w[1]-w[0] for w in self.gvec])
		n_seg  = np.round(incres/strides)
		print(n_seg,end='\t')
		strides = strides/n_seg
		for l in range(3):
			self._data = np.transpose(self._data,[1,2,0])
			if n_seg[l] > 1:
				li = self.gvec[l]
				new_li = np.linspace(li[0],li[-1],n_seg[l]*(li.size-1)+1)
				self._data = np.asarray([[Akima1DInterpolator(self.gvec[l],line)(new_li) for line in mat] for mat in self._data])
				self.gvec[l] = new_li

	def finite_diff(self):
		"""
		"""
		strides = np.asarray([w[1]-w[0] for w in self.gvec])
		self._grad_data = np.gradient(self._data, *strides)
		print("gradient grids generated",end='\t')
		hess_data = np.empty((3,3)+self._data.shape, dtype='d')
		for k, g_k in enumerate(self._grad_data):
			tmp_g = np.gradient(g_k, *strides)
			for l, g_kl in enumerate(tmp_g):
				hess_data[k,l,:,:] = g_kl[:,:]
		print("hessian grids generated")
		self._hess_data = {}
		for i in range(3):
			self._hess_data[(i,i)] = hess_data[i,i]
			for j in range(i+1,3):
				self._hess_data[(i,j)] = 0.5*(hess_data[i,j]+hess_data[j,i])

	def check_laplace(self, trace_free=True, tol=0.1):
		"""
		Checking if self._hess_data satisfies laplace equation
		parameters:
			trace_free:: Whether to fix the hessian diagonals to satisfy Laplace eq. or not. 
				If trace_free, self._hess_data would be automatically trace-free
			tol:: Tolerance in nonzero laplacian. Unit [V]/[L]^2, [L] is the length unit of this electrode
		"""
		laplace = np.asarray([self._hess_data[(i,i)] for i in range(3)]).sum(axis=0)
		nbad = (laplace > tol).sum()
		if nbad > 0:
			laplace_ = laplace[1:-1,1:-1,1:-1]
			nbad_ = (laplace_ > tol).sum()
			print("Laplace eq. check: totally %d bad points, %d inside the bulk."%(nbad, nbad_))
			# print("Largest residue is", abs(laplace).max(),". Mean(abs) =",abs(laplace).mean())
			# print("Chopping off the boundaries,","Largest residue is", abs(laplace_).max(),". Mean(abs) =",abs(laplace_).mean())
			if trace_free:
				laplace /= 3.
				for i in range(3):
					self._hess_data[(i,i)] -= laplace
			return (nbad, nbad_)
		else:
			return (0,0)

	def interpolate(self):
		"""
		Interpolate the grid data
		Future dev: Should either 3D Akima or 3D spline be developed, replace the RegularGridInterpolator method here
		"""
		# bounds_error=False, fill_value=None enables extrapolation
		self._interpolant = RegularGridInterpolator(self.gvec, self._data, method='linear', bounds_error=False, fill_value=None)
		self._grad_interpolants = [RegularGridInterpolator(self.gvec, g, method='linear', bounds_error=False, fill_value=None) for g in self._grad_data]
		self._hess_interpolants = {}
		for k in self._hess_data.keys():
			self._hess_interpolants[k] = RegularGridInterpolator(self.gvec, self._hess_data[k], method='linear', bounds_error=False, fill_value=None)

	# def output_data(self):
	# 	return self._data #, 'grad':self._grad_data, 'hessian':self._hess_data}

	def pot(self, pos):
		return self._interpolant(pos)

	def grad(self, pos):
		# try:
		return np.asarray([gi(pos) for gi in self._grad_interpolants]).ravel()
		# except ValueError:
		# 	print(pos,[self.gvec[l][[0,-1]] for l in range(3)])
		# 	sys.exit()

	def hessian(self, pos):
		hess = np.empty((3,3),dtype='d')
		for ky in self._hess_interpolants.keys():
			i,j = ky
			if i==j:
				hess[i,j] = (self._hess_interpolants[ky](pos)).ravel()[0]
			else:
				hess[i,j] = (self._hess_interpolants[ky](pos)).ravel()[0]
				hess[j,i] = hess[i,j]
		return hess

	def hessdiag(self, pos):
		return np.asarray([self._hess_interpolants[(l,l)](pos) for l in range(3)]).ravel()

	def output_hessdata(self):
		return self._hess_data

class SqElectrode:
	"""
	SqElectrode: Squares-shaped Electrode
	"""
	def __init__(self, location, derivatives):
		"""
		location is a 2-element list of the form
		[ (xmin, xmax), (ymin, ymax) ]

		axes_permutation is an integer.
		0 (default): normal surface trap. z-axis lies along the plane
		of the trap
		1: trap is in the x-y plane. z axis is vertical
		"""

		xmin, xmax = sorted(location[0])
		ymin, ymax = sorted(location[1])

		(self.x1, self.y1) = (xmin, ymin)
		(self.x2, self.y2) = (xmax, ymax)

		self.multipole_expansion_dict = {} # keys are expansion points. elements are dictionaries of multipole expansions

		self.derivatives = derivatives

	def pot(self, r):
		'''
		The solid angle for an arbitary rectangle oriented along the grid is calculated by
		Gotoh, et al, Nucl. Inst. Meth., 96, 3

		The solid angle is calculated from the current electrode, plus any additional electrodes
		that are electrically connected to the current electrode. This allows you to join electrodes
		on the trap, or to make more complicated electrode geometries than just rectangles.
		'''
		x, y, z = r
		solid_angle = self.derivatives['phi0'](self.x1, self.x2, self.y1, self.y2, x, y, z) / (2*np.pi)

		return solid_angle

	def grad(self, r):
		'''
		gradient of the solid angle at the observation point
		'''
		x, y, z = r
		keys = ['ddx', 'ddy', 'ddz']
		grad = np.array([self.derivatives[key](self.x1, self.x2, self.y1, self.y2, x, y, z)
						 for key in kforbiddeneys]) / (2*np.pi)
		return grad

	def hessian(self, r):
		'''
		Hessian matrix at the observation point
		'''

		x, y, z = r
		hessian = np.zeros((3,3))
		
		hessian[0,0] = self.derivatives['d2dx2'](self.x1, self.x2, self.y1, self.y2, x, y, z)
		hessian[1,1] = self.derivatives['d2dy2'](self.x1, self.x2, self.y1, self.y2, x, y, z)
		hessian[2,2] = self.derivatives['d2dz2'](self.x1, self.x2, self.y1, self.y2, x, y, z)
		
		hessian[0,1] = hessian[1,0] = self.derivatives['d2dxdy'](self.x1, self.x2, self.y1, self.y2, x, y, z)
		hessian[0,2] = hessian[2,0] = self.derivatives['d2dxdz'](self.x1, self.x2, self.y1, self.y2, x, y, z)
		hessian[1,2] = hessian[2,1] = self.derivatives['d2dydz'](self.x1, self.x2, self.y1, self.y2, x, y, z)

		hessian = hessian / (2*np.pi)
		return  hessian

	def third_order_derivatives(self, r):
		'''
		We're not going to include all of them here, probably.
		'''
		keys = ['d3dz3', 'd3dxdz2','d3dydz2']
		x,y,z = r
		third_derivatives = np.array([self.derivatives[key](self.x1, self.x2, self.y1, self.y2, x, y, z)
						 for key in keys]) / (2*np.pi)
		return third_derivatives

	def fourth_order_derivatives(self, r):
		keys = ['d4dz4', 'd4dx2dz2', 'd4dy2dz2']
		x, y, z = r
		fourth_derivatives = np.array([self.derivatives[key](self.x1, self.x2, self.y1, self.y2, x, y, z)
						 for key in keys]) / (2*np.pi)
		return fourth_derivatives
