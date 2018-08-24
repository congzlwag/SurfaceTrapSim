from __future__ import print_function
# import struct
import numpy as np
import numdifftools as nd
# import pdb
# import pickle
# import cPickle as pickle #Instead for python 2
import h5py
from hdf5storage import loadmat, savemat
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import eigh, inv, norm
from scipy.interpolate import Akima1DInterpolator, RectBivariateSpline
from matplotlib import pyplot as plt
# import csv
from tqdm import tqdm
from time import clock
import sys

from electrode import RRPESElectrode, Electrode
# from multipole_expansion import MultipoleExpander

class World:
	'''
	A General, Brand New World in SI-Unit
	Axis convention in consistance with class:Electrode
		x: perpendicular to the trap axis and parallel to the surface trap
		y: perpendicular to the surface trap
		z: axial
	attributes:
		__scale:: the typical length in meter. Length unit in the code is self.scale meter(s)
		dc_electrode_list:: a list of (name, electrode) s of dc electrodes
		rf_electrode_list:: a list of (name, electrode) s of rf electrodes
	'''
	qe = 1.602176487e-19 # Charge of electron in Coulomb
	amu = 1.66054e-27    # Atomic Mass Unit in kg
	def __init__(self, ionA, omega_rf, scale=1):
		self.rf_electrode_list = []
		self.dc_electrode_list = []
		self.omega_rf = omega_rf
		self.m = ionA * World.amu
		self.__scale = scale
		self._pseudopot_factor = World.qe**2/(4*self.m*(omega_rf*self.__scale)**2)
		self.bounds = None # if no boundary, then None

	def add_electrode(self, e, name, kind, volt):
		"""
		Add an electrode to the World. Name it with `name.
		If kind == 'rf', then add this electrode to the rf electrode dict
		as well as to the general electrode dict
		"""
		e.volt = volt
		if kind=='dc':
			self.dc_electrode_list.append((name,e))
		if kind=='rf':
			self.rf_electrode_list.append((name,e))

	def compute_dc_potential(self, r):
		v = 0
		for nam, e in self.dc_electrode_list:
			v += e.compute_potential(r)
		return v # the potential energy is automatically electron volts

	def compute_dc_field(self, r):
		E = np.zeros(3)
		for nam, e in self.dc_electrode_list:
			E += e.compute_electric_field(r)
		return E
	
	def compute_dc_hessian(self, r):
		hess = np.zeros((3,3))
		for nam, e in self.dc_electrode_list:
			hess += e.compute_hessian(r)
		return hess

	def compute_rf_field(self, r):
		"""
		Just add up the electric field due to all the rf electrodes
		not the gradient of pseudopotential
		"""
		E = np.zeros((3))
		for nam, e in self.rf_electrode_list:
			try:
				E += e.compute_electric_field(r)
			except ValueError:
				print(e.compute_electric_field(r))
				sys.exit()
		return E

	def check_bound(self):
		if self.bounds is None:
			boundless = True
			lbs = []
			ubs = []
		else:
			boundless = False
			lbs = [list(self.bounds[:,0])]
			ubs = [list(self.bounds[:,1])]
		for nm, elec in self.dc_electrode_list+self.rf_electrode_list:
			bds = elec.get_region_bounds()
			if bds is not None:
				boundless = False
				lbs.append(bds[0])
				ubs.append(bds[1])
		if not boundless:
			lbs = np.max(np.asarray(lbs),0)
			ubs = np.min(np.asarray(ubs),0)
			self.bounds = np.asarray([lbs,ubs]).T
		else:
			self.bounds = None

	def compute_rf_null(self, z, xy0=(0,0), onyz=False, bounds=None):
		if bounds is None:
			self.check_bound()
			if self.bounds is None:
				print("Cannot carry out RF null searching without bounds")
				return
			else:
				bounds = self.bounds
		if onyz: # x=0 required
			fo = lambda y: sum(self.compute_rf_field(np.array([xy0[0],y,z]))**2)
			ym = minimize_scalar(fo, bounds=tuple(bounds[1]), method="Bounded")
			if ym.success:
				ymy = ym.x
			else:
				print("@ z=%.3fmm Optimization Failed:"%z, ym.message, '. Returning initial value')
				ymy = xy0[1]
				yi = np.linspace(yi[0],yi[-1],30)
				plt.plot(yi, [fo(yy) for yy in yi], label="%.3f"%z)
				plt.xlabel("y/mm")
				plt.ylabel(r"$E_{RF}^2/\mathrm{(V^2mm^{-2})}$")
				plt.title("RF null @x = 0")
			# if ym.success:
			# 	plt.plot([ymy],[ym.fun],'x')
			# else:
				# plt.legend(title="z/mm")
				plt.show()
			return np.array([xy0[0],ymy,z])
		else:
			foo= lambda xy: norm(self.compute_rf_field(np.array([xy[0],xy[1],z])))
			if self.bounds is None:
				xym = minimize(foo, xy0)
			else:
				xym = minimize(foo, xy0, bounds=bounds[:2])
			if xym.success:
				return np.array([xym.x[0],xym.x[1],z])
			else:
				print("Optimization Failure", xym.message, '. Returning initial value')
				return np.array([xy0[0],xy0[1],z])

	def compute_pseudopot(self, r):
		"""pseudopotential in Joule"""
		return self._pseudopot_factor*sum(self.compute_rf_field(r)**2)

	def compute_pseudopot_frequencies(self, r):
		'''
		This is only valid if xp, yp, zp is the trapping position. Return frequency (i.e. 2*pi*omega)
		'''
		hessdiag = nd.Hessdiag(self.compute_pseudopot,step=1e-6)(r)/(self.__scale**2)
		'''
		Now d2Udx2 has units of J/m^2. Then w = sqrt(d2Udx2/(mass)) has units of angular frequency
		'''
		return np.sqrt(abs(hessdiag)/self.m)/(2*np.pi)

	def compute_dc_potential_frequencies(self, r):
		'''
		As always, this is valid only at the trapping position. Return frequency (not angular frequency)
		'''
		
		H = self.compute_dc_hessian(r)
		
		hessdiag, eigvec = eigh(H)
		hessdiag /= self.__scale**2

		# # hessdiag = nd.Hessdiag( self.compute_total_dc_potential )(r)
		# d2Udx2 = ev_to_joule*hessdiag[0]
		# d2Udy2 = ev_to_joule*hessdiag[1]
		# d2Udz2 = ev_to_joule*hessdiag[2]
		# fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
		# fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
		# fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
		return np.sqrt(abs(hessdiag)/self.m)/(2*np.pi), eigvec

	def compute_full_potential(self, r):
		return self.compute_pseudopot(r) + self.compute_total_dc_potential(r)

	# def compute_full_potential_frequencies(self, r):
	#   '''
	#   As always, this is valid only at the trapping position. Return frequency (not angular frequency)
	#   '''
		
	#   joule_to_ev = 6.24150934e18 # conversion factor to take joules -> eV
	#   ev_to_joule = 1.60217657e-19
	#   m = 6.64215568e-26 # 40 amu in kg
			
	#   H = nd.Hessian(self.compute_full_potential)(r)   
		
	#   freq = np.linalg.eig(H)
	#   hessdiag = freq[0]
	#   eigvec = freq[1]

	#   # hessdiag = nd.Hessdiag( self.compute_total_dc_potential )(r)
	#   d2Udx2 = ev_to_joule*hessdiag[0]
	#   d2Udy2 = ev_to_joule*hessdiag[1]
	#   d2Udz2 = ev_to_joule*hessdiag[2]
	#   fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
	#   fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
	#   fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
	#   return [fx, fy, fz], eigvec

	def construct_multipole_arr(self, position, ctrl_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], ctrl_electrodes='alldc', r0=1):
		"""
		return: The matrix that maps DC voltages on `ctrl_electrodes to `ctrl_multipoles at `position
				Its shape is (len(ctrl_multipoles), len(ctrl_electrodes))
		"""
		self.ctrl_multipoles = ctrl_multipoles
		if isinstance(ctrl_electrodes,str) and ctrl_electrodes=='alldc':
			ctrl_electrodes = range(len(self.dc_electrode_list))

		multipole_arr = np.empty((len(ctrl_multipoles), len(ctrl_electrodes)),'d')
		for i, j in enumerate(ctrl_electrodes):

			nam, elec = self.dc_electrode_list[j]
			elec.expand_in_multipoles(position, ctrl_multipoles, r0)

			multipole_arr[:,i] = [elec.multipole_dict[multipole] for multipole in ctrl_multipoles]
		return multipole_arr

	def multipole_control_vdc_arr(self, position, ctrl_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], ctrl_electrodes = 'alldc', cost_Q='id', r0=1):
		"""
		return: The matrix that controls voltages on `ctrl_electrodes in order for `ctrl_multipoles at `position.
				Its shape is (len(self.dc_electrodes), len(ctrl_multipoles)) with those not multipole-ctrl electrodes padded with 0
		"""
		alle = (isinstance(ctrl_electrodes,str) and ctrl_electrodes=='alldc')
		if alle:
			ctrl_electrodes = range(len(self.dc_electrode_list))
		multipole_arr = self.construct_multipole_arr(position, ctrl_multipoles, ctrl_electrodes, r0)
		
		n_elec = len(ctrl_electrodes)
		if isinstance(cost_Q,str) and cost_Q=='id':
			cost_Q = np.identity(n_elec)
		assert cost_Q.shape==(n_elec, n_elec)
		cost_P = inv(cost_Q)
		A = multipole_arr
		assert A.ndim==2 and cost_P.ndim==2

		PAt = np.dot(cost_P,A.T)
		kernel = inv(np.dot(A,PAt)) # maps a multipole set-up to the half of the Lagrange multipliers
		voltage_arr = np.dot(PAt,kernel)      # maps a multipole set-up to the optimized voltage set-up
		
		if not alle: #padding
			for i in range(len(self.dc_electrode_list)):
				if not i in ctrl_electrodes:
					voltage_arr = np.insert(voltage_arr, i, np.zeros(len(ctrl_multipoles)),axis=0)
		return voltage_arr

	# def calculate_multipoles(self, multipole_list):
	# 	multipoles = {}
	# 	for mult in multipole_list:      
	# 		multipoles[mult] = sum([elec.multipole_dict[mult]*elec.volt for ky, elec in enumerate(self.electrode_dict)])            
	# 	return multipoles
