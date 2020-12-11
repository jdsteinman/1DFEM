# 1D Finite Element Code
import numpy as np
import numexpr as ne
import math
from scipy.linalg import inv
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

# Intialize and Preallocate
Ne = 10   # Number of Elements
N = Ne+1  # Number of Nodes

f = '(pi**2-1)*e**(-t)*sin(pi*x)'
u_guess = 'sin(pi*x)'
global_expr = {'pi': math.pi, 'e': math.e}

K = np.zeros((N,N))    # N by N global stiffness matrix
M = np.zeros((N,N))    # N by N global mass matrix
k_local = np.zeros((2,2))    # local stiffness matrix
m_local = np.zeros((2,2))    # local mass matrix

g_w = np.array([1., 1.])  # Gaussian quadrature weights
g_xi = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])  # Gaussian quadrature points in xi space

psi_1 = '(1-xi)/2'
psi_2 = '(1+xi)/2'
dpsi_1 = -1/2
dpsi_2 = 1/2
dpsi = np.array([-1/2, 1/2])
psi_1_quad = np.array(ne.evaluate(psi_1, local_dict={'xi': g_xi}))
psi_2_quad = np.array(ne.evaluate(psi_2, local_dict={'xi': g_xi}))
psi_quad = np.vstack((psi_1_quad, psi_2_quad))

# mesh
x = np.linspace(0,1,N)
h = np.diff(x)
dbc = np.array(([0, 0], [N-1,0]))

# time discretization
dt = 1/1000
time = np.arange(0,1+dt,dt)

# connectivity
map_lg = np.arange(Ne)   
map_lg = np.vstack((map_lg, map_lg+1)).T

# Start Method
for k in range(Ne):      # loop through elements
    for l in range(2):   # loop over nodes on element k
        for m in range(2):   # mass and stiffness
            m_local[l][m] = h[k]/2*np.sum(psi_quad[l,:]*psi_quad[m,:]*g_w)
            k_local[l][m] = 2/h[k]*dpsi[l]*dpsi[m]*2
    
    # Sum local terms
    for l in range(2):
        global_node = map_lg[k][l]
        for m in range(2):
            global_node2 = map_lg[k][m]
            M[global_node][global_node2] += m_local[l][m]
            K[global_node][global_node2] += k_local[l][m]

# Invert Matrices
B = 1/dt*M + K
B_inv = inv(B)
B_inv_copy = B_inv.copy()
B_inv_M = B_inv@M
B_inv_M_copy = B_inv_M.copy()

# Boundary Conditions
for i in range(N):
    if(i in dbc[:,0]):
        B_inv_M[i,:] = np.zeros(N)
        B_inv_M[:,i] = np.zeros(N)
        B_inv[i,:] = np.zeros(N)
        B_inv[:,i] = np.zeros(N)
        B_inv[i,i] = 1

open('F.txt', 'w').close()
open('u.txt', 'w').close()
open('A.txt', 'w').close()

#Solve in time
u = ne.evaluate(u_guess, local_dict = {'x':x}, global_dict=global_expr)
iter = 0

for t in time:
    iter += 1
    # Find F vector
    F = np.zeros(N)

    # Gaussian Quadrature
    for k in range(Ne):
        g_x = h[k]/2*(g_xi+1) + x[k]  # map from xi to x space
        for l in range(2):
            f_quad = ne.evaluate(f, local_dict={'x':g_x, 't': t}, global_dict=global_expr)
            f_local = h[k]/2*np.sum(g_w*f_quad*psi_quad[l,:])
            global_node = map_lg[k][l]
            F[global_node] += f_local

    # Boundary Conditions
    for j in range(N):
        if(j in dbc[:,0]):
            idx=np.where(dbc[:,0]==j)
            F[j] = dbc[idx,1]/dt
            for i in range(N):
                if(i != j):
                    F[i] -= B_inv_copy[i,j] * dbc[idx,1]/dt 
   
    with open('F.txt', 'a') as file:
        np.savetxt(file, F.reshape((1,N)), fmt='%4.2f', delimiter=' ')      
        
    with open('u.txt', 'a') as file:
        np.savetxt(file, u.reshape((1,N)), fmt='%4.2f', delimiter=' ')       

    u_new = 1/dt * B_inv_M @ u + B_inv @ F
    u = u_new

fig, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(x, u)
ax[0].grid(True)
ax[0].set_xlabel('x')
ax[0].set_ylabel('u')
ax[0].set_title('FEM solution at t=1')

x_real = np.linspace(0,1,100)
u_real = ne.evaluate('e**-t*sin(pi*x)', local_dict = {'x': x_real, 't': 1}, global_dict=global_expr)
ax[1].plot(x_real, u_real)
ax[1].grid(True)
ax[1].set_xlabel('x')
ax[1].set_ylabel('u')
ax[1].set_title('Analytical solution at t=1')
plt.show()
