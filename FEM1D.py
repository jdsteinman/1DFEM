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
global_expr = {'pi': math.pi, 'e': math.e}

K = np.zeros((N,N))    # N by N global stiffness matrix
M = np.zeros((N,N))    # N by N global mass matrix
k_local = np.zeros((2,2))    # local stiffness matrix
m_local = np.zeros((2,2))    # local mass matrix

g_w = np.array([1, 1])  # Gaussian quadrature weights
g_p = np.array([-1/math.sqrt(3), 1/math.sqrt(3)])  # Gaussian quadrature points

phi_1 = '(1-xi)/2'
phi_2 = '(1+xi)/2'
dphi_1 = -1/2
dphi_2 = 1/2
dphi = np.array([-1/2, 1/2])
# f_quad = np.array(ne.evaluate(f, local_dict = {'t':0, 'x': g_p}, global_dict=global_expr)) # TODO: Think this is wrong
phi_1_quad = np.array(ne.evaluate(phi_1, local_dict={'xi': g_p}))
phi_2_quad = np.array(ne.evaluate(phi_2, local_dict={'xi': g_p}))
phi_quad = np.vstack((phi_1_quad, phi_2_quad))


# mesh
x = np.linspace(0,1,N)
h = np.diff(x)
dbc = np.array(([0, 0], [N-1,0]))

# time discretization
dt = 1/551
time = np.arange(0,1+dt,dt)
print('final time' + str(time[-1]))

# connectivity
map_lg = np.arange(Ne)   
map_lg = np.vstack((map_lg, map_lg+1)).T

# Start Method
for k in range(Ne):      # loop through elements
    for l in range(2):   # loop over nodes on element k
        for m in range(2):   # mass and stiffness
            m_local[l][m] = h[k]/2*np.sum(np.dot(np.dot(phi_quad[l,:], phi_quad[m,:]), g_w))
            k_local[l][m] = 2/h[k]*dphi[l]*dphi[m]*2
    
    # Sum local terms
    for l in range(2):
        global_node = map_lg[k][l]
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
                M[j,i] = 0
                M[i,j] = 0
                K[j,i] = 0
                K[i,j] = 0
             
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
    x_m = x[0:-1]+h/2 # interval midpoints
    f_m = np.array(ne.evaluate(f, local_dict = {'t':t, 'x': x_m}, global_dict=global_expr))  # function vals at midpoints
    for k in range(Ne):
        for l in range(2):  
            f_local = h[k]*f_m[k]*1/2  #dx*f(x_m)*phi(x_m)
            global_node = map_lg[k][l]
            F[global_node] += f_local

    # Boundary Conditions
    for i in range(N):
        if(i in dbc[:,0]):
            idx = np.where(dbc[:,0]==i) 
            for j in range(N):
                if(j != i):
                    F[j] -= K[j,i]*dbc[idx,1]-2/dt*M[j,i]*dbc[idx,1] #TODO: Not sure if this is right         
                    continue
    a = dt*M_inv_K@u
    b = dt*M_inv@F
    u_new = u - a + b
    u = u_new

fig, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(x, u)
ax[0].grid(True)
ax[0].set_xlabel('x')
ax[0].set_ylabel('u')
ax[0].set_title('FEM solution at t=1')

x_real = np.linspace(0,1,100)
u_real = ne.evaluate(f, local_dict = {'x': x_real, 't': 1}, global_dict=global_expr)
ax[1].plot(x_real, u_real)
ax[1].grid(True)
ax[1].set_xlabel('x')
ax[1].set_ylabel('u')
ax[1].set_title('Analytical solution at t=1')
plt.show()
