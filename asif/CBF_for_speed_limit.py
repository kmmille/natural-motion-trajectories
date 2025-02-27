#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mark

Applies to in-plane dynamics

"""

import numpy as np
import os, sys, inspect
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) # add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities')) # add utilities to path
from parameters import SystemParameters
import gurobipy as gp
from gurobipy import GRB

class ASIF(SystemParameters):
    def __init__(self):

        ########################## Set Parameters #############################

        # Define ASIF parameters
        self.alpha_coefficient = 0.5  # lower values give more of a "buffer"

        # Define safety set and backup set parameters
        self.K1_s = 2*self.mean_motion # slope of safety boundary for speed limit constraint (must be >= 2n)
        self.K2_s = 0.2*0.925 # # max allowable speed at origin (must be > eta_b)
        # self.K = self.K1_s

        ################### Do Not Modify Below this Line #####################
        self.Fmax = self.max_available_thrust

        # Define CWH Dynamics
        self.A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [3*self.mean_motion**2, 0, 0, 2*self.mean_motion],
                  [0, 0, -2*self.mean_motion, 0]])
        self.B = np.array([[0, 0],
                  [0, 0],
                  [1/self.mass_chaser, 0],
                  [0, 1/self.mass_chaser]])

        self.Hs = np.array( [[ 2*self.K1_s**2,           0,   0,    0],
                              [           0, 2*self.K1_s**2,   0,    0],
                              [           0,           0,  -2,    0],
                              [           0,           0,   0,   -2] ] )

    ##########################################################################
    def main(self, x0, u_des):
        """
        Parameters
        ----------
        x : numpy array with 6 elements
            x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
            where x,y,z are hill frame coordinates
                u : 3x1 numpy array
        u_des = [[Fx], [Fy], [Fz]]
            desired control input

        Returns
        -------
        u : 3x1 numpy array
            u = [[Fx], [Fy], [Fz]]
            returned control input
        """

        ######################## CBF Algorithm ###############################

        Fx_des = u_des[0,0]
        Fy_des = u_des[1,0]

        # Reduce dimension of x since we are only looking in-plane
        x = np.array([ [x0[0]], [x0[1]], [x0[3]], [x0[4]] ] )

        # Calculate Subregulation Map using: hdot = sigma + eta*u
        sigma = np.matmul(self.grad_hs(x), np.matmul(self.A,x))

        eta = np.matmul(self.grad_hs(x), self.B)
        etax = eta[0]
        etay = eta[1]

        # Barrier constraint hdot + alpha(h(x)) >= 0
        alpha_hs = self.alpha(self.h_s(x))
        # print("alpha(hs(x)) = ", alpha_hs)

        # Initialize states
        Fx = []
        Fy = []
        dist_out_of_bounds = []

        m = gp.Model("CBF")

        # Define variables at each of the tau timesteps
        Fx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -self.Fmax, ub = self.Fmax, name="Fx" ))
        Fy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -self.Fmax, ub = self.Fmax, name="Fy" ))
        dist_out_of_bounds.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = 1000*self.Fmax, name="DOB" ))

        print('lenth of thing',len(Fx))

        m.update()

        # Set boundary conditions
        b = alpha_hs + sigma
        m.addConstr( etax*Fx[0] + etay*Fy[0] + dist_out_of_bounds[0] >= -b  , "BC")
        # m.addConstr( dist_out_of_bounds[0] == 0 )

        # Set Objective
        obj = Fx[0]*Fx[0] + Fy[0]*Fy[0] - 2*Fx_des*Fx[0] - 2*Fy_des*Fy[0] + 100000*dist_out_of_bounds[0]


        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam( 'OutputFlag', False )

        # Optimize and report on results
        m.optimize()


        # Save desired trajectory
        # self.xstar = np.zeros([6, tau])
        self.ustar = np.zeros([3, 1])

        # for t in range(tau):
            # self.xstar[0,t] = m.getVarByName("sx"+str(t)).x
            # self.xstar[1,t] = m.getVarByName("sy"+str(t)).x
            # self.xstar[3,t] = m.getVarByName("vx"+str(t)).x
            # self.xstar[4,t] = m.getVarByName("vy"+str(t)).x
        self.ustar[0,0] = m.getVarByName("Fx").x
        self.ustar[1,0] = m.getVarByName("Fy").x
        # DOB = m.getVarByName("DOB").x
        # print("DOB = ", DOB)

        self.ustar[2,0] = u_des[2,0]
            # self.ustar[2,t] = m.getVarByName("Fz"+str(t)).x


        return self.ustar

    ##########################################################################
    def h_s(self, x):
        """
        h_s(x) >= 0 defines the set of all "safe states". The goal of the ASIF
        is to ensure that this constraint remains satisfied

        """
        x = x.flatten()
        r2 = x[0]**2 + x[1]**2
        # print("h = ", self.K2_s**2 + (self.K1_s**2)*(r2) + 2*self.K1_s*self.K2_s*np.sqrt(r2) - (x[2]**2 + x[3]**2) )
        return self.K2_s**2 + (self.K1_s**2)*(r2) + 2*self.K1_s*self.K2_s*np.sqrt(r2) - (x[2]**2 + x[3]**2)

    ##########################################################################
    def grad_hs(self, x):
        x = x.flatten()

        ghs = np.matmul( self.Hs, x )
        rn1 = 1/np.sqrt(x[0]**2 + x[1]**2) # 1/r
        ghs[0] = ghs[0] + 2*self.K1_s*self.K2_s*rn1*x[0]
        ghs[1] = ghs[1] + 2*self.K1_s*self.K2_s*rn1*x[1]
        # print(ghs)

        return ghs

    ##########################################################################
    def alpha(self, x):
        # print("x = ", x)
        return  self.alpha_coefficient*x**3
