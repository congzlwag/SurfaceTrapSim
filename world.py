from __future__ import print_function
import numpy as np
import numdifftools as nd
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import eigh, inv, norm
from scipy.constants import e as qe # Charge of electron in Coulomb
from matplotlib import pyplot as plt
from tqdm import tqdm

from .SH import funcSHexp

class World:
	'''
A General, Brand New World in SI-Unit

Axis convention in consistance with <class Electrode>
	z: axial
	* It doesn't matter whether z is parallel or vertical to the surface or not

Attributes:
	__scale	:: the typical length in meter. Length unit in the code is self.__scale meter(s)
	omega_rf:: the RF ANGULAR frequency
	m 		:: the mass a single ion
	bounds 	:: the boundaies of this world
	dc_electrode_list	:: a list of (name, electrode) s of dc electrodes
	rf_electrode_list	:: a list of (name, electrode) s of rf electrodes
	electrode_dict		:: dictionary that electrode_dict[name] = ("dc" or "rf", electrode)
	_pseudopot_factor	:: the factor in front of the pseudopotential

Methods:
	'''
	amu = 1.66054e-27    # Atomic Mass Unit in kg
	def __init__(self, ionA, omega_rf, scale=1):
		"""
__init__(self, ionA, omega_rf, scale=1):
	ionA:		mass number of the ion
	omega_rf: 	the RF ANGULAR frequency
	scale	: 	the typical length in meter. Length unit in the code is self.__scale meter(s)
		"""
		self.electrode_dict = {}
		self.rf_electrode_list = []
		self.dc_electrode_list = []
		self.omega_rf = omega_rf
		self.m = ionA * World.amu
		self.__scale = scale
		self._pseudopot_factor = qe**2/(4*self.m*(omega_rf**2))/(scale**2)
		self.bounds = None # if no boundary, then None

	def add_electrode(self, e, name, kind, volt):
		"""
		Add an electrode to the World. Name it with `name.
		If kind == 'rf', then add this electrode to the rf electrode dict
		as well as to the general electrode dict
		"""
		e.volt = volt
		self.electrode_dict[name] = (kind, e)
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
			E += e.compute_electric_field(r)
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
				yi = np.linspace(bounds[1,0],bounds[1,1],30)
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

	def compute_pseudopot_hessian_atNULL(self, r):
		"""This only valid when r is a RF null!"""
		hess = np.zeros((3,3))
		for nm, e in self.rf_electrode_list:
			hess += e.compute_hessian(r)
		return 2*self._pseudopot_factor*(np.dot(hess, hess.T))

	def compute_pseudopot_frequencies(self, r):
		'''
		This is only valid if xp, yp, zp is the trapping position. Return frequency (i.e. omega/(2*pi))
		'''
		hessdiag = nd.Hessdiag(self.compute_pseudopot,step=1e-6)(r)/(self.__scale**2)
		'''
		Now d2Udx2 has units of J/m^2. Then w = sqrt(d2Udx2/(mass)) has units of angular frequency
		'''
		return np.sqrt(qe*abs(hessdiag)/self.m)/(2*np.pi), hessdiag>0

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
		return np.sqrt(qe*abs(hessdiag)/self.m)/(2*np.pi), eigvec, hessdiag>0

	def compute_full_potential(self, r):
		return self.compute_pseudopot(r) + self.compute_dc_potential(r)

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

	def local_multipole_arr(self, position, loc_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], ctrl_electrodes='alldc', r0=1):
		"""
local_multipole_arr(self, position, loc_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], ctrl_electrodes='alldc', r0=1)
Parameters:
	position		:: a (3,) array indicating the position of interest
	loc_multipoles	:: a list of multipoles that are of interest at this position
	ctrl_electrodes :: a list of the INDICES of dc electrodes of interest
	
returns the matrix, shaped (len(loc_multipoles), len(ctrl_electrodes)), that maps DC voltages on `ctrl_electrodes to `loc_multipoles at `position
		"""
		self.loc_multipoles = loc_multipoles
		if isinstance(ctrl_electrodes,str) and ctrl_electrodes=='alldc':
			ctrl_electrodes = range(len(self.dc_electrode_list))

		multipole_arr = np.empty((len(loc_multipoles), len(ctrl_electrodes)),'d')
		for i, j in enumerate(ctrl_electrodes):

			nam, elec = self.dc_electrode_list[j]
			elec.expand_in_multipoles(position, loc_multipoles, r0)

			multipole_arr[:,i] = [elec.multipole_dict[multipole] for multipole in loc_multipoles]
		return multipole_arr

	def multipole_control_vdc_arr(self, pos_ctrl_mults, ctrl_electrodes='alldc', cost_Q='id', r0=1):
		"""
multipole_control_vdc_arr(self, pos_ctrl_mults, ctrl_electrodes='alldc', cost_Q='id', r0=1):
Parameters:
	position	:: a (3,) array indicating the position of interest
	pos_ctrl_mults	:: a list of (position, list of controlled local multipoles) pairs or a single pair
	ctrl_electrodes :: a list of the INDICES of dc electrodes to be multipole-controlled
	costQ  		:: the positive definite matrix Q in the cost function

return: The matrix, shaped (len(self.dc_electrodes), n_mult), controls DC voltages on ctrl_electrodes for pos_ctrl_mults.
	n_mult = sum([len(pos_ctrl_mult[1]) for pos_ctrl_mult in pos_ctrl_mults])
	Rows that correspond to electrodes that are not multipole-controlled are padded with 0
		"""
		alle = (isinstance(ctrl_electrodes,str) and ctrl_electrodes=='alldc')
		if alle:
			ctrl_electrodes = range(len(self.dc_electrode_list))

		# Support inputing a single (position, list of controlled local multipoles) pair
		if isinstance(pos_ctrl_mults, tuple) and len(pos_ctrl_mults)==2:
			pos_ctrl_mults = [pos_ctrl_mults]

		n_mult = sum([len(pos_ctrl_mult[1]) for pos_ctrl_mult in pos_ctrl_mults])
		n_elec = len(ctrl_electrodes)
		if n_mult > n_elec:
			raise ValueError("Number of multipoles %d exceeds number of controlled electrodes %d"%(n_mult, n_elec))

		multipole_arr = np.empty((n_mult, n_elec), 'd')

		pt = 0
		for pos, ctrl_mult in pos_ctrl_mults:
			multipole_arr[pt:pt+len(ctrl_mult),:] = self.local_multipole_arr(pos, ctrl_mult, ctrl_electrodes, r0)
			pt += len(ctrl_mult)

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
					voltage_arr = np.insert(voltage_arr, i, np.zeros(n_mult), axis=0)
		return voltage_arr

	def set_volts(self, voltages, kind):
		elist = self.dc_electrode_list if kind=='dc' else self.rf_electrode_list
		assert len(voltages)==len(elist)
		for i, nm_e in enumerate(elist):
			nm_e[1].volt = voltages[i]

	def fit_field_hess(self, pos, h_grid, n_grid=5, order=2):
		if n_grid%2==0:
			n_grid += 1
		gi = h_grid*(np.arange(n_grid) - n_grid//2)
		f, res = funcSHexp(self.compute_full_potential, pos, pos[0]+gi, pos[1]+gi, pos[2]+gi, 3)
		print(res)
		pot = f[0]
		efield = np.array([f[2],f[3],-f[1]])
		quad = np.array([6*f[7], f[4], 12*f[8], -6*f[6], -6*f[5]])
		hess = np.empty((3,3),'d')
		hess[0,0] = quad[0]-quad[1]
		hess[1,1] = -quad[0]-quad[1]
		hess[2,2] = 2*quad[1]
		hess[0,1] = hess[1,0] = 0.5*quad[2]
		hess[0,2] = hess[2,0] = 0.5*quad[4]
		hess[1,2] = hess[2,1] = 0.5*quad[3]
		return pot, efield, hess