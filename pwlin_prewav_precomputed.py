from P1_FEM import *
from numpy import *
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
import os

"""
2018 Lukas Herrmann, SAM, ETH Zurich

this code outputs pw linear coeff of Gaussian random fields 
with given precision operator L^2, where L is a second order, 
symmetric elliptic precision operator with periodoc BCs on D=(0,1) 
the GRF is discretized by pw linear prewavelets
"""


def GRF_prewav_to_hat_precomputed(Lmatrices, theta, kappa, alpha, level, y):
	"""
	Computes GRF function values expanded in prewavelets

	:param Lmatrices: cholesky factors
	:param theta: positive scale parameter
	:param kappa: positive scale parameter
	:param alpha: positive scale parameter
	:param level: maximal level
	:param y: i.i.d. Gaussian white noise
	:return: scaled function values of GRF
	"""
	
	l = level
	N = 2**(l+2)

	assert(len(y) == N)

	mesh = gen_mesh((0, 1), N, "equidistant")
	coeffs = 1
	coeffs2 = lambda x: kappa**2*(1 + 0.5*sin(2*pi*x))

	y_t = block_chol_M_precomputed(Lmatrices, l, y)

	RHS = prewav_to_hat(l, y_t)

	A0 = assemble_stiffness_periodicBC(mesh, coeffs)
	M0 = assemble_mass_periodicBC(mesh, coeffs2)

	Z = zeros(N+1)

	S = (A0 + M0) 

	Z[0:N] = spsolve(S, RHS)

	if alpha >= 3:
		# activate for alpha == 4
		M1 = assemble_mass_periodicBC(mesh, coeffs)
		RHS2 = M1.dot( Z[0:N] )
		Z[0:N] = spsolve(S,RHS2)

	Z[N] = Z[0]

	return theta*Z


def block_chol_M_precomputed(Lmatrices,l,y):
	assert(len(y)==2**(l+2))

	res = zeros(len(y))

	for k in range(l+1):

		# cases k=0, k=1 are special cases
		if k == 0:
			
			L = Lmatrices[k]
			y_tilde = y[:2**(k+2)]
			res[:2**(k+2)] = L.dot( y_tilde )
		if k >= 1:
			L = Lmatrices[k]
			y_tilde = y[2**(k+1):2**(k+2)]
			res[2**(k+1):2**(k+2)] = L.dot( y_tilde )
	return res


def prewav_to_hat(level, psi):
	"""
	Outputs coefficients in one scale hat function basis from prewavelet coefficients
	Input:
		level : maximal level
		psi : prewavelet coefficients
	"""
	l = level
	if l == 0:
		return psi

	phi = psi
	
	for k in range(l):
		P = wav_to_hat(k+1)		
		phi[:2**(k+2+1)] = spsolve(P, phi[:2**(k+2+1)])

	return phi


def wav_to_hat(level):
	"""
	Outputs basis transformation matrix for one block corresponding to level
	Input:
		level : current prewavelet level
	"""

	k = level
	assert(k > 0)	
	
	# size of block mass matrix
	N = 2**(k+1)
	M = 2**(k+2)
	V = zeros(3*N + 5*N)
	I = zeros(3*N + 5*N)
	J = zeros(3*N + 5*N)

	#hat function interpolation part
	c = array([0.5, 1., 0.5])
		
	# first two and last two line are special cases
	#first line
	V[0:2] = c[1:3]
	V[2] = c[0]
	I[:3] = array([0,0,0])
	J[:3] = array([0,1,M-1])

	for kk in range(N-2):
		k_tilde = kk+1
		V[3*k_tilde:3*(k_tilde+1)] = c
		I[3*k_tilde:3*(k_tilde+1)] = array([k_tilde,k_tilde,k_tilde])
		J[3*k_tilde:3*(k_tilde+1)] = array([-1+2*k_tilde,2*k_tilde,2*k_tilde+1])

	#last line
	V[3*(N-1):3*N] = c
	I[3*(N-1):3*N] = array([1,1,1]) * (N-1)
	J[3*(N-1):3*N] = array([M-3, M-2, M-1])	
	
	#wavelet part
	c = array([0.5,-3.,5.,-3.,0.5])
		
	# first last line are special cases
	#first line
	V[3*N:3*N+4] = c[1:5]
	V[3*N+4:3*N+5] = c[0]
	I[3*N:3*N+5] = N + array([0,0,0,0,0])
	J[3*N:3*N+5] = array([0,1,2,3,M-1])

	for kk in range(N-2):
		k_tilde = kk+1
		V[3*N + 5*k_tilde:3*N + 5*(k_tilde+1)] = c
		I[3*N + 5*k_tilde:3*N + 5*(k_tilde+1)] = N + array([k_tilde,k_tilde,k_tilde,k_tilde,k_tilde])
		J[3*N + 5*k_tilde:3*N + 5*(k_tilde+1)] = array([-1 + 2*k_tilde, 2*k_tilde,1+2*k_tilde,2*k_tilde+2,2*k_tilde+3])

	#last line
	V[3*N + 5*(N-1):3*N + 5*N-2] = c[:3]
	V[3*N + 5*N-2:3*N + 5*N] = c[3:5]
	I[3*N + 5*(N-1):3*N + 5*N] = N + array([1,1,1,1,1]) * (N-1)
	J[3*N + 5*(N-1):3*N + 5*N] = array([M-3, M-2, M-1, 0, 1])

	P = sparse.csc_matrix((V,(I,J)),shape=(M,M))

	return P


def prewav_to_hat_coeffs(level):
	k = level
	assert(k > 0)	
	
	# size of block mass matrix
	N = 2**(k+2)
	M = N/2
	V = zeros(3*M + 5*M)
	I = zeros(3*M + 5*M)
	J = zeros(3*M + 5*M)

	c1 = array([0.5,0.5])
	c2 = array([-3, -3])
	c3 = array([0.5, 5., 0.5])

	for kk in range(M):
		#contribution of hat function coeffs of level-1
		V[3*kk] = 1
		I[3*kk] = 2*kk
		J[3*kk] = kk
		V[3*kk+1:3*(kk+1)] = c1
		I[3*kk+1:3*(kk+1)] = array([2*kk+1, 2*kk+1])
		J[3*kk+1:3*(kk+1)] = mod(array([kk, kk+1]), M)

		#contribution of prewavelet coeffs
		V[3*M +5*kk: 3*M +5*kk +2] = c2
		I[3*M +5*kk: 3*M +5*kk +2] = 2*kk
		J[3*M +5*kk: 3*M +5*kk +2] = M + mod(array([kk, kk+1]), M)
		V[3*M +5*kk+2: 3*M +5*(kk +1)] = c3
		I[3*M +5*kk+2: 3*M +5*(kk +1)] = array([2*kk+1, 2*kk+1, 2*kk+1])
		J[3*M +5*kk+2: 3*M +5*(kk +1)] = M + mod(array([kk, kk+1, kk+2, ]), M)

	P = sparse.csc_matrix((V, (I, J)), shape=(N, N))

	return P 


def compute_chol_M(l):
	"""
	Computes and saves mass matrix
	"""

	for k in range(l+1):
		L_matrix_dir = "L_matrices"
		if not os.path.exists(L_matrix_dir): os.mkdir(L_matrix_dir)

		# cases k=0, k=1 are special cases
		if k==0:
			L_matrix_name = L_matrix_dir + "/L_matrix%d.npy"%(k)
			h = 2**(-(k+2))
			M = array([[4, 1, 0, 1], [1, 4, 1, 0], [0, 1, 4, 1], [1, 0, 1, 4]]) * h/6.
			L = linalg.cholesky( M )
			save(L_matrix_name, L)
			
		if k==1:
			L_matrix_name = L_matrix_dir + "/L_matrix%d.npy"%(k)
			h = 2**(-(k+2))
			M = array([[108, 20, -4, 20],[20, 108, 20, -4], [-4, 20, 108, 20], [20, -4, 20, 108]]) * h/6.
			L = linalg.cholesky( M )
			save(L_matrix_name, L)

		if k>=2:
			L_matrix_name = L_matrix_dir + "/L_matrix%d.npz"%(k)
			h = 2**(-(k+2))

			# size of block mass matrix
			N = 2**(k+1)
			V = zeros(5*N)
			I = zeros(5*N)
			J = zeros(5*N)

			c = array([-2,20,108,20,-2]) * h/6.
		
			# first two and last two line are special cases
			#first line
			V[:3] = c[2:5]
			V[3:5] = c[:2]
			I[:5] = array([0, 0, 0, 0, 0])
			J[:5] = array([0, 1, 2, N-2, N-1])

			#second line
			V[5:5+4] = c[1:5]
			V[5+4] = c[0]
			I[5:5+5] = array([1, 1, 1, 1, 1])
			J[5:5+5] = array([0, 1, 2, 3, N-1])

			for kk in range(N-4):
				k_tilde = kk+2
				V[5*k_tilde:5*(k_tilde+1)] = c
				I[5*k_tilde:5*(k_tilde+1)] = array([k_tilde, k_tilde, k_tilde, k_tilde, k_tilde])
				J[5*k_tilde:5*(k_tilde+1)] = array([-2 + k_tilde, -1+k_tilde, k_tilde, k_tilde+1, k_tilde+2])
		
			#second to last line
			V[5*(N-2):5*(N-1)-1] = c[:4]
			V[5*(N-1)-1] = c[4]
			I[5*(N-2):5*(N-1)] = array([1, 1, 1, 1, 1]) * (N-2)
			J[5*(N-2):5*(N-1)] = array([N-4, N-3, N-2, N-1,0])

			#last line
			V[5*(N-1):5*N-2] = c[:3]
			V[5*N-2:5*N] = c[3:5]
			I[5*(N-1):5*N] = array([1, 1, 1, 1, 1]) * (N-1)
			J[5*(N-1):5*N] = array([N-3, N-2, N-1, 0, 1])
		
		
			M = sparse.csc_matrix((V,(I,J)),shape=(N,N))

			#compute chol of M = L L^T
			factor = cholesky( M )
			L = factor.L()
			sparse.save_npz(L_matrix_name, L)
	
	return 0


def hat_to_prewav_new( level ):
	
	k = level
	assert(k > 0)	
	
	# size of block mass matrix
	N = 2**(k+1)
	M = 2**(k+2)
	V = zeros(5*N)
	I = zeros(5*N)
	J = zeros(5*N)

	c = array([0.5, -3., 5., -3., 0.5])
		
	# first two and last two line are special cases
	#first line
	V[:3] = c[2:5]
	V[3:5] = c[:2]
	I[:5] = array([0, 0, 0, 0, 0])
	J[:5] = array([0, 1, 2, M-2, M-1])

	for kk in range(N-2):
		k_tilde = kk+1
		V[5*k_tilde:5*(k_tilde+1)] = c
		I[5*k_tilde:5*(k_tilde+1)] = array([k_tilde, k_tilde, k_tilde, k_tilde, k_tilde])
		J[5*k_tilde:5*(k_tilde+1)] = array([-2 + 2*k_tilde,-1+2*k_tilde,2*k_tilde,2*k_tilde+1,2*k_tilde+2])

	#last line
	V[5*(N-1):5*N-2] = c[:3]
	V[5*N-2:5*N] = c[3:5]
	I[5*(N-1):5*N] = array([1, 1, 1, 1, 1]) * (N-1)
	J[5*(N-1):5*N] = array([M-3, M-2, M-1, 0, 1])

	P = sparse.csc_matrix((V, (I, J)), shape=(N, M))

	return P

# interpolate hat functions
def hat_to_hat_interpolate(level):
	k = level
	assert(k > 0)	
	
	# size of block mass matrix
	N = 2**(k+1)
	M = 2**(k+2)
	V = zeros(3*N)
	I = zeros(3*N)
	J = zeros(3*N)

	c = array([0.5, 1., 0.5])
		
	# first two and last two line are special cases
	#first line
	V[:1] = c[1:2]
	V[3] = c[0]
	I[:3] = array([0, 0, 0])
	J[:5] = array([0, 1, M-1])

	for kk in range(N-2):
		k_tilde = kk+1
		V[3*k_tilde:3*(k_tilde+1)] = c
		I[3*k_tilde:3*(k_tilde+1)] = array([k_tilde, k_tilde, k_tilde, k_tilde, k_tilde])
		J[3*k_tilde:3*(k_tilde+1)] = array([-1+2*k_tilde, 2*k_tilde, 2*k_tilde+1])

	#last line
	V[3*(N-1):3*N-2] = c[:2]
	V[3*N-1] = c[2]
	I[3*(N-1):3*N] = array([1, 1, 1]) * (N-1)
	J[3*(N-1):3*N] = array([M-2, M-1, 0])

	I = sparse.csc_matrix((V, (I, J)), shape=(N, M))

	return I


