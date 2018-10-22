from __future__ import print_function
__all__ = ['gridSHexp', 'funcSHexp']


import numpy as np
from scipy.special import lpmn
from scipy.linalg import lstsq

def gridSHexp(V, xri,yri,zri, order):
	"""
V:				gridded values
xri, yri, zri:	relative grid vectors
order:			max l in the spherical harmonic expansion
	"""
	assert xri.ndim==1 and yri.ndim==1 and zri.ndim==1
	n_samp = xri.size*yri.size*zri.size
	assert n_samp >= (order+1)**2
	Y,X,Z = np.meshgrid(yri,xri,zri)

	for A in [X,Y,Z]:
		A.shape = (-1,) #flattening
	W = V.reshape(-1,)

	Phi = np.arctan2(Y,X)
	# print(Phi)
	Theta = np.arctan2((X**2+Y**2)**0.5, Z)
	R = (X**2+Y**2+Z**2)**0.5
	r0 = R.max()**0.5
	R /= r0

	Q = np.empty((n_samp, (order+1)**2),'d')
	Q[:,0] = 1
	# We don't expand on scipy.special.sph_harm
	# Instead, expand on P_l^m(cos\theta) * (cos(m\phi), sin(m\phi))^T
	MPhi = (np.atleast_2d(Phi).T)*np.arange(1,order+1)
	# print(MPhi)
	cos_sin_MPhi = np.stack((np.cos(MPhi),np.sin(MPhi)),axis=-1).reshape(n_samp,-1)
	# print(cos_sin_MPhi)
	for p in range(n_samp):
		# print(X[p], Y[p], Z[p])
		aslegd = lpmn(order, order, np.cos(Theta[p]))[0]
		# aslegd is now an upper triangle order+1 by order+1 mat
		# aslegd[m,l] = P_l^m(Theta[p]) if m<=l else 0
		pt_col = 1
		for l in range(1,order+1):
			r_pl_theta = (R[p]**l)*aslegd[:l+1,l]
			# print("\t",r_pl_theta)
			Q[p, pt_col] = r_pl_theta[0]
			pt_col += 1
			Q[p, pt_col:pt_col+2*l] = np.repeat(r_pl_theta[1:],2)*cos_sin_MPhi[p,:2*l]
			pt_col += 2*l
		assert pt_col==(order+1)**2

	# print(Q)
	# print(W)
	fit = lstsq(Q, W)
	rms = (fit[1]/n_samp)**0.5

	rescale = [1]
	for l in range(1,order+1):
		rescale = rescale + [r0**l]*(2*l+1)
	rescale = np.array(rescale)
	# print(rescale)

	# return fit[0], rms
	return fit[0]/rescale, rms


def funcSHexp(func, rc, xi, yi, zi, order):
	W = np.empty((xi.size,yi.size,zi.size),'d')
	for i, x in enumerate(xi):
		for j,y in enumerate(yi):
			for k,z in enumerate(zi):
				W[i,j,k] = func([x,y,z])
	return gridSHexp(W, xi-rc[0], yi-rc[1], zi-rc[2], order)

if __name__ == '__main__':
	gi = np.arange(-2,3)
	f,r = funcSHexp(lambda r: -0.5*r[1]**2+0.5*r[0]**2, [0,0,0], gi,gi,gi,3)
	print("U1",f)
	f,r = funcSHexp(lambda r: r[2]**2-0.5*r[1]**2-0.5*r[0]**2, [0,0,0], gi,gi,gi,3)
	print("U2",f)
	f,r = funcSHexp(lambda r: 0.5*r[1]*r[0], [0,0,0], gi,gi,gi,3)
	print("U3",f)
	f,r = funcSHexp(lambda r: 0.5*r[2]*r[1], [0,0,0], gi,gi,gi,3)
	print("U4",f)
	f,r = funcSHexp(lambda r: 0.5*r[2]*r[0], [0,0,0], gi,gi,gi,3)
	print("U5",f)
