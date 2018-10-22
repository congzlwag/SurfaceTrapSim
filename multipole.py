from __future__ import print_function
import numpy as np
from .SH import gridSHexp

multipole_convention = 'Littich'
# Multipole convention
# 		Da: Da An's convention, used in cct gapless
# 		Littich: Convention in Gebhard Littich thesis, p.39

__all__ = ['MultExp']

class MultExp:
	"""
	"""
	_expand_pos = None
	_taylor_dict = {}
	multipole_dict = {}

	def get_expand_pos(self):
		return self._expand_pos

	def expand_in_multipoles(self, r, controlled_multipoles=['C','Ex','Ey','Ez','U1','U2','U3','U4','U5'], r0=1):
		"""
		Obtain the multipole expansion for the potential due to the electrode at the observation point.
		"""
		# first, make sure taylor expansion at position r is done
		self.expand_potential(r)

		for cm in controlled_multipoles:
			if not cm in self.multipole_dict.keys():
				self.multipole_dict[cm] = self.get_multipole(cm, r0)

	def expand_potential(self, r):
		"""
		Numerically expand the potential due to the electrode to second order as a taylor series
		around the obersvation point r = [x, y, z]

		self._taylor_dict is a dictionary containing the terms of the expansion. e.g.
		self._taylor_dict['x^2'] = (1/2)d^2 \phi/dx^2
		"""
		if np.allclose(self._expand_pos,r,1e-10) and len(self._taylor_dict)>0: #reuse previous results
			return

		# Otherwise we need to gather the _taylor_dict for this NEW position
		# Clear out multipole_dict
		self.multipole_dict = {}
		# Gather
		self._expand_pos = r
		self._taylor_dict = {}
		self._taylor_dict['C'] = self.pot(r)
		self._taylor_dict['x'], self._taylor_dict['y'], self._taylor_dict['z'] = self.grad(r)
		# second derivatives
		hessian = self.hessian(r)
		if multipole_convention=="Da":
			self._taylor_dict['x^2'] = 0.25*hessian[0,0]
			self._taylor_dict['y^2'] = 0.25*hessian[1,1]
			self._taylor_dict['z^2'] = 0.25*hessian[2,2]
			self._taylor_dict['xy'] = 0.25*hessian[0,1]
			self._taylor_dict['xz'] = 0.25*hessian[0,2]
			self._taylor_dict['zy'] = 0.25*hessian[1,2]
		elif multipole_convention=="Littich":
			self._taylor_dict['x^2'] = 0.5*hessian[0,0]
			self._taylor_dict['y^2'] = 0.5*hessian[1,1]
			self._taylor_dict['z^2'] = 0.5*hessian[2,2]
			self._taylor_dict['xy'] = hessian[0,1]
			self._taylor_dict['xz'] = hessian[0,2]
			self._taylor_dict['zy'] = hessian[1,2]

		# # higher order stuff
		# self._taylor_dict['z^3'], self._taylor_dict['xz^2'], self._taylor_dict['yz^2'] = self.third_order_derivatives(r)
		# self._taylor_dict['z^4'], self._taylor_dict['x^2z^2'], self._taylor_dict['y^2z^2'] = self.fourth_order_derivatives(r)