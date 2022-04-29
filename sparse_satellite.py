import numpy as np
import matplotlib as plt
import cvxpy as cp
from scipy.signal import cont2discrete


## System Model

# Physical parameters
mean_motion = 0.001027 # [rad/s]
mass_chaser = 140 # [kg]

max_available_thrust = 2 # [N]

controller_sample_rate = .2 # [Hz]
controller_sample_period = 1/controller_sample_rate # [s]

filter_sample_rate = 1 # [Hz]
filter_sample_period = 1/filter_sample_rate # [s]

# Plant matrices

A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [3*mean_motion**2, 0, 0, 0, 2*mean_motion, 0],
              [0, 0, 0, -2*mean_motion, 0, 0],
              [0, 0, -mean_motion**2, 0, 0, 0]])
b = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1/mass_chaser, 0, 0],
              [0, 1/mass_chaser, 0],
              [0, 0, 1/mass_chaser]])
d = len(b)
dims = b.shape


C = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
D = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])

# Initial state
x = 1000 # [m]
x0 = np.array([[x],  # x
               [0],  # y
               [0],  # z
               [0],  # xdot
               [0],  # ydot
               [0]]) # zdot

# Horizon length
T = 5

## Time Discretization

# Discretization size
n = 10# grid size
h = T/n # discretization interval

# System discretization
Ad, bd, Cd, Dd, hd = cont2discrete((A,b,C,D), h)
Phi = np.zeros((d,n*dims[1]))
v = bd
Phi[:,(n-1)*dims[1]:n*dims[1]] = v
for j in range(1,n):
    v = np.matmul(Ad,v)
    Phi[:,(n-j-1)*dims[1]:(n-j)*dims[1]] = v

Zeta = np.matmul((Ad**n),x0)

## Convex Optimization using CVX

x = cp.Variable((n*3, 1))
cost = cp.norm(x, 1)
prob = cp.Problem(cp.Minimize(cost), [cp.matmul(Phi, x) == Zeta])
prob.solve()

print(x.value)
