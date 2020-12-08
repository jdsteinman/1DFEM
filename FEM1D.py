# 1D Finite Element Code
import numpy as np
import scipy

# Intialize and Preallocate
Ne = 10   # Number of Elements
N = Ne+1  # Number of Nodes
Ne_d = 2  # Number of Dirichlet Nodes
J_k = 1   # TODO Jacobian of kth element

F = zeros(N,1)  # N by 1 global force vector
K = zeros(N)    # N by N global stiffness matrix
f_local = zeros(2,1)  # local force on element nodes
k_local = zeros(2)    # local stiffness matrix

#


# Need a map
map_lg = {}   # connectivity

# Start Method
for k in range(Ne):      # loop through elements
    # Integration
    for l in range(3)  # loop over nodes on element k
        # f_local = ... # Numerical integration on parent element
        for m in range(2)   # stiffness
            # k_local = ... # Numerical integration 
    
    # Sum local terms
    for l in range(3)
        g = map_lg  # TODO map[N,l]
        #F(g) = F(g) + f_local(l)
        for m in range(2)

            # g2 = map_lg #map[N,m]
            # K(g, g2) = K(g, g2) + k_local(l,m)


# Boundary Conditions
for i in range(N)
    if(true) # TODO if node is dirichlet node
        for j in range(NN)
            if(j != i)
                F[j] += K(j,i)*1
                K[j,i] = 0
                K[i,j] = 0
             
            F(i) = 1 # assign dirichlet val
            K(i,i) = 1


# Solve Ku=f
u = linsolve(K,F)



