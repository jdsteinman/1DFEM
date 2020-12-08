# 1D Finite Element Code
import numpy as np
import numexpr as ne
import math
import scipy

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
phi_2_quad = np.array(ne.evaluate(phi_1, local_dict={'xi': g_p}))
phi_quad = np.vstack((phi_1_quad, phi_2_quad))

# mesh
x = np.linspace(0,1,N)
h = np.diff(x)
dbc = np.array(([0, 0], [N-1,0]))

# time discretization
dt = 1/551
t = np.arange(0,1,dt)

# connectivity
map_lg = np.arange(Ne)   
map_lg = np.vstack((map_lg, map_lg+1)).T

# Boundary Conditions

# Start Method
for k in range(Ne):      # loop through elements
    for l in range(2):   # loop over nodes on element k
        f_local[l] = h[k]/2*np.sum(np.dot(np.dot(f_quad, phi_quad[l,:]), g_w))
        for m in range(2):   # mass and stiffness
            m_local[l][m] = h[k]/2*np.sum(np.dot(np.dot(phi_quad[l,:], phi_quad[m,:]), g_w))
            k_local[l][m] = 2/h[k]*dphi[l]*dphi[m]
    
    # Sum local terms
    for l in range(2):
        global_node = map_lg[k][l]
        F[global_node] += f_local[l]
        for m in range(2):
            global_node2 = map_lg[k][m]
            M[global_node][global_node2] += m_local[l][m]
            K[global_node][global_node2] += k_local[l][m]

print(F)
print(M)
np.savetxt('out.txt', K, delimiter=" ")

# Boundary Conditions
for i in range(N):
    if(i in dbc[0,:]): 
        for j in range(N):
            if(j != i):
                F[j] -= K[j,i]*dbc[i,1]
                K[j,i] = 0
                K[i,j] = 0
             
            F[i] =  dbc[i,1]   # assign dirichlet val
            K[i,i] = 1

print(F)
np.savetxt('out.txt', K, fmt="%4.2f", delimiter=" ")

# Solve Ku=f
# u = linsolve(K,F)

def integ_f(f_exp, g_x, g_w, h):
    x = g_x
    F = np.array(ne.evaluate(f_exp))
    prod = np.dot(F, )
    F = h/2*np.dot(g_w, F)
    return F
