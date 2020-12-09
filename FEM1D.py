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
Ne_d = 2  # Number of Dirichlet Nodes

f = '(pi**2-1)*e**(-t)*sin(pi*x)'
global_expr = {'pi': math.pi, 'e': math.e}

F = np.zeros(N)  # N by 1 global force vector
K = np.zeros((N,N))    # N by N global stiffness matrix
M = np.zeros((N,N))    # N by N global mass matrix
f_local = np.zeros(2)  # local force on element nodes
k_local = np.zeros((2,2))    # local stiffness matrix
m_local = np.zeros((2,2))    # local mass matrix

g_w = np.array([1, 1])  # Gaussian quadrature weights
g_p = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])  # Gaussian quadrature points

phi_1 = '(1-xi)/2'
phi_2 = '(1+xi)/2'
dphi_1 = -1/2
dphi_2 = 1/2
dphi = np.array([-1/2, 1/2])
f_quad = np.array(ne.evaluate(f, local_dict = {'t':0, 'x': g_p}, global_dict=global_expr))
phi_1_quad = np.array(ne.evaluate(phi_1, local_dict={'xi': g_p}))
phi_2_quad = np.array(ne.evaluate(phi_2, local_dict={'xi': g_p}))
phi_quad = np.vstack((phi_1_quad, phi_2_quad))


# mesh
x = np.linspace(0,1,N)
h = np.diff(x)
dbc = np.array(([0, 0], [N-1,0]))

# time discretization
dt = 1/551
time = np.arange(0,1,dt)

# connectivity
map_lg = np.arange(Ne)   
map_lg = np.vstack((map_lg, map_lg+1)).T

# Boundary Conditions

# Start Method
for k in range(Ne):      # loop through elements
    for l in range(2):   # loop over nodes on element k
        # f_local[l] = h[k]/2*np.sum(np.dot(np.dot(f_quad, phi_quad[l,:]), g_w))
        for m in range(2):   # mass and stiffness
            m_local[l][m] = h[k]/2*np.sum(np.dot(np.dot(phi_quad[l,:], phi_quad[m,:]), g_w))
            k_local[l][m] = 2/h[k]*dphi[l]*dphi[m]*2
    
    # Sum local terms
    for l in range(2):
        global_node = map_lg[k][l]
        # F[global_node] += f_local[l]
        for m in range(2):
            global_node2 = map_lg[k][m]
            M[global_node][global_node2] += m_local[l][m]
            K[global_node][global_node2] += k_local[l][m]

# Boundary Conditions
for i in range(N):
    if(i in dbc[:,0]):
        idx = np.where(dbc[:,0]==i) 
        for j in range(N):
            if(j != i):
                F[j] -= K[j,i]*dbc[idx,1]-2/dt*M[j,i]*dbc[idx,1] #TODO: Not sure if this is right         
                M[j,i] = 0
                M[i,j] = 0
                K[j,i] = 0
                K[i,j] = 0
             
            F[i] =  dbc[idx,1]   # assign dirichlet val
            M[i,i] = 1
            K[i,i] = 1

np.savetxt('out.txt', M, fmt='%4.2f', delimiter=' ')

# Invert Matrices
M_inv = inv(M)
M_inv_K = M_inv@K

# Solve in time
u = np.zeros(N)

for t in time:
    # Find F vector
    F = np.zeros(N)

    # Midpoint rule TODO: how to do Gaussian Quadrature?
    x_m = x[0:-1]+h/2
    f_m = np.array(ne.evaluate(f, local_dict = {'t':t, 'x': x_m}, global_dict=global_expr))
    for k in range(Ne):
        for l in range(2):  
            f_local = h[k]*f_m[k]*1/2
            global_node = map_lg[k][l]
            F[global_node] += f_local

    a = dt*M_inv_K@u
    b = dt*M_inv@F
    u_new = u - a + b
    u = u_new

fig, ax = plt.subplots()
ax.plot(x, u)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Solution at t=1')
plt.show()
