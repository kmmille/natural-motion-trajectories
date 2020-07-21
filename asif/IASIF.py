#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mark Mote (marklmote@gmail.com)

A template for creating RTA filters, passes desired input through unaltered 

The main loop will call the ASIF's "main" class and expect 
a control signal of appropriate size and type to be returned 

In order for this script to work correctly with the main script, avoid 
changing the name of the "ASIF" class, or the inputs and outputs of 
the "main" function.  

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
        self.zero_input = np.zeros([3,1])
        
        # Flags 
        self.f_use_heuristic = True # Apply heuristic that will speed up computation (but hasn't formally been proven to work) 
        
        # Define Backup controller parameters 
        self.T_backup = 2100 # [s] length of time in backup trajectory horizon 
        self.Nsteps = 70     # number steps in horizon of backup trajectory 
        self.kappa =  3     # higher values of this make actuation stonger  
        
        # Define safety set and backup set parameters 
        self.eta_b = 0.1    # acceptable error magnitude from NMT plane for reachability constraint
        self.K1_s = 2.0*self.mean_motion # slope of safety boundary for speed limit constraint (must be >= 2n)
        self.K2_s = 0.2 # 2*self.eta_b # max allowable speed at origin (must be > eta_b)
        
        self.timevec = np.linspace(0, self.T_backup, self.Nsteps)
        self.dt = self.timevec[1]-self.timevec[0]
        
        # Define in-plane CWH Dynamics 
        self.A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [3*self.mean_motion**2, 0, 0, 2*self.mean_motion],
                  [0, 0, -2*self.mean_motion, 0]])
        self.B = np.array([[0, 0], 
                           [0, 0],
                           [1/self.mass_chaser, 0],
                           [0, 1/self.mass_chaser]])
        
        # Define matrix for gradient of hb(x)
        n = self.mean_motion
        n2 = n**2 
        self.Hb = np.array( [[ -8*n2,       0,  0, -4*n],
                             [     0, -0.5*n2,  n,    0], 
                             [     0,       n, -2,    0], 
                             [  -4*n,       0,  0,   -2] ] )
        self.Hs = np.array( [[ 2*self.K1_s**2,           0,   0,    0],
                              [           0, 2*self.K1_s**2,   0,    0], 
                              [           0,           0,  -2,    0], 
                              [           0,           0,   0,   -2] ] )
        
        self.phi  = np.zeros([4, self.Nsteps])
        self.Dphi = np.zeros([4, 4, self.Nsteps])
        self.initialize_Dphi() 
        
    ##########################################################################    
    def main(self, x0, u_des):
        """
        Parameters
        ----------
        x : numpy array with 6 elements 
            x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
            where x,y,z are hill frame coordinates 
        u: 3x1 numpy array
        u_des = [[Fx], [Fy], [Fz]]
            desired control input 

        Returns
        -------
        u : 3x1 numpy array
            u = [[Fx], [Fy], [Fz]]
            returned control input  
        """
        
        # Cut out "z" dimension in x0 
        x0 = np.array([x0[0], x0[1], x0[3], x0[4]])
              
        # Heuristic: decide whether to run asif
        run_asif = True 
        if self.f_use_heuristic: 
            safe_dist = 0.5 
            rmag = np.sqrt(x0[0]**2 + x0[1]**2)
            vmag = np.sqrt(x0[2]**2 + x0[3]**2)
            if (vmag <= self.K1_s*rmag + self.K2_s - safe_dist) :
                run_asif = False
        
        if run_asif:
            # Integrate finite time trajectory under backup control law to initialize "self.phi" array
            self.integrate(x0)
            
            try: 
                ################### Set up optimization program ######################
                Fx_des = u_des[0,0]
                Fy_des = u_des[1,0]
                
                 # Initialize states 
                Fx = [] 
                Fy = [] 
                dist_out_of_bounds1 = [] 
                dist_out_of_bounds2 = [] 
                
                m = gp.Model("IASIF")
                
                # Define variables at each of the tau timesteps  
                Fx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -self.max_available_thrust, ub = self.max_available_thrust, name="Fx" )) 
                Fy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -self.max_available_thrust, ub = self.max_available_thrust, name="Fy" )) 
                dist_out_of_bounds1.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = 1000*self.max_available_thrust, name="DOB1" )) 
                dist_out_of_bounds2.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = 1000*self.max_available_thrust, name="DOB2" )) 
                    
                m.update()
                
                ######################################################################
                
                Ax0 = np.matmul(self.A, x0.reshape(4,1))
        
                ################ Backwards reachability constraint ################### 
                # Define coefficients 
                ghb = self.grad_hb(self.phi[:,-1])
                d = np.matmul(ghb, np.matmul(self.Dphi[:,:,-1], Ax0)) + self.alpha( self.h_b( self.phi[:,-1] ) )
                c = np.matmul(ghb, np.matmul(self.Dphi[:,:,-1], self.B ))
                
                # Round to prevent Gurobi warnings 
                if np.abs(c[0])<1e-13: 
                    c[0]=1e-13*np.sign(c[0])
                if np.abs(c[1])<1e-13: 
                    c[1]=1e-13*np.sign(c[0])
                if np.abs(d[0])<1e-13: 
                    d[0]=1e-13*np.sign(c[0])
                
                # Add constraint 
                m.addConstr( c[0]*Fx[0] + c[1]*Fy[0] + dist_out_of_bounds1[0] >= -d[0]  , "BRC")
                
                
                ################### Set invariance constraints ####################### 
                for i in range(0,self.Nsteps):
                    # print(i)
                    # Define coefficients 
                    ghs = self.grad_hs(self.phi[:,i])
                    d = np.matmul(ghs, np.matmul(self.Dphi[:,:,i], Ax0)) + self.alpha( self.h_s( self.phi[:,i] ) )
                    c = np.matmul(ghs, np.matmul(self.Dphi[:,:,i], self.B ))
                    
                    # Round to prevent Gurobi warnings 
                    if np.abs(c[0])<1e-13: 
                        c[0]=1e-13*np.sign(c[0])
                    if np.abs(c[1])<1e-13: 
                        c[1]=1e-13*np.sign(c[0])
                    if np.abs(d[0])<1e-13: 
                        d[0]=1e-13*np.sign(c[0])
                        
                    # Add constraint 
                    m.addConstr( c[0]*Fx[0] + c[1]*Fy[0] + dist_out_of_bounds2[0] >= -d[0]  , "BC"+str(i))
                    # m.addConstr( c[0]*Fx[0] + c[1]*Fy[0]  >= -d[0]  , "BC"+str(i))
        
                
                
                
                ################### Solve optimization program! ######################
                # Set Objective
                obj = Fx[0]*Fx[0] + Fy[0]*Fy[0] - 2*Fx_des*Fx[0] - 2*Fy_des*Fy[0] + 10000*dist_out_of_bounds1[0]+10000*dist_out_of_bounds2[0]
        
                m.setObjective(obj, GRB.MINIMIZE)
                m.setParam( 'OutputFlag', False )
                
                # Optimize and report on results 
                m.optimize()
                
                # Pull in results 
                self.ustar = np.zeros([3, 1])
                
                self.ustar[0,0] = m.getVarByName("Fx").x
                self.ustar[1,0] = m.getVarByName("Fy").x
                self.ustar[2,0] = u_des[2,0] # force input in z direction equal to desired        
            except: 
                print("Optimization failed! You may be starting in an unsafe region... Reverting to backup controller")
                ub = self.u_b(x0)
                self.ustar = np.array([ [ub[0,0]], [ub[1,0]], [u_des[2,0]] ])
            
            return self.ustar 
        else: 
            return u_des
   
    ########################################################################    
    def h_s(self, x):
        """
        h_s(x) >= 0 defines the set of all "safe states". The goal of the ASIF 
        is to ensure that this constraint remains satisfied 
        
        """
        r2 = x[0]**2 + x[1]**2
        return self.K2_s**2 + (self.K1_s**2)*(r2) + self.K1_s*self.K2_s*np.sqrt(r2) - (x[2]**2 + x[3]**2) 

    # ##########################################################################        
    def grad_hs(self, x):
        ghs = np.matmul( self.Hs, x )
        rn1 = 1/np.sqrt(x[0]**2 + x[1]**2) # 1/r 
        ghs[0] = ghs[0] + 2*self.K1_s*self.K2_s*rn1*x[0] 
        ghs[1] = ghs[1] + 2*self.K1_s*self.K2_s*rn1*x[1] 
        # print(ghs)
        return ghs 

    ##########################################################################  
    def h_b(self, x):
        """
        h_b(x) >= 0 defines the "terminal set".
        
        """
        sx = x[0]
        sy = x[1]
        vx = x[2]
        vy = x[3]
        n = self.mean_motion
        
        return self.eta_b**2 - vx**2 - vy**2 - 4*(n**2)*sx**2 - 0.25*(n**2)*sy**2 - 4*n*sx*vy + n*sy*vx 
    
    ##########################################################################        
    def grad_hb(self, x):
        return np.matmul( self.Hb, x )
    
    ##########################################################################            
    def u_b(self, x0):
                
        e1 = x0[2]-(self.mean_motion/2)*x0[1]
        e2 = x0[3]+(2*self.mean_motion)*x0[0]
        
        umax = self.max_available_thrust
        
        u = np.array([ [umax*np.tanh((-1.5*self.mean_motion*e2 - self.kappa*e1)/umax)],
                       [umax*np.tanh((-self.kappa*e2)/umax)]])
        
        return u
        
    ##########################################################################        
    def alpha(self, x ): 
        return 10*x**3 
    
    ##########################################################################        
    def xdot(self, x, u): 
        return (np.matmul(self.A, x.reshape(4,1)) + np.matmul(self.B, u)).reshape(4)
    
    ##########################################################################    
    def integrate(self, x0):
        # integrate backup controller to get phi, Dphi 
        
        self.phi[:,0] = x0 
        # self.Dphi[:,:,0] = np.eye(4) 
        
        for i in range(1,self.Nsteps): 
            # Dynamics 
            self.phi[:,i] = self.phi[:,i-1] + self.xdot(self.phi[:,i-1], self.u_b(self.phi[:,i-1]))*self.dt 
            
            # # Sensitivity 
            # self.Dphi[:,:,i] = self.Dphi[:,:,i-1] + np.matmul(self.A, self.Dphi[:,:,i-1])*self.dt
        
    ##########################################################################
    def initialize_Dphi(self):
        # integrate sensitivity matrix to get Dphi array
        
        self.Dphi[:,:,0] = np.eye(4) 
        
        for i in range(1,self.Nsteps): 
            # Sensitivity 
            self.Dphi[:,:,i] = self.Dphi[:,:,i-1] + np.matmul(self.A, self.Dphi[:,:,i-1])*self.dt
        
        
        
        
        
        
        
        
        
        
        
        
        
        