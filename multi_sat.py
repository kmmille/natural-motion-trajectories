#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kris
modified from MIQP_to_NMT.py by mark mote

"""

# import numpy as np
# import matplotlib.pyplot as plt
# import os, sys, inspect
# sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
# import gurobipy as gp
# from gurobipy import GRB


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rand
import sys
from parameters import SystemParameters
from ClohessyWiltshire import ClohessyWiltshire
from mpl_toolkits.mplot3d import Axes3D
import gurobipy as gp
from gurobipy import GRB




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Import Active Set Invariance Filter (ASIF) (aka RTA mechanism)
from asif.CBF_for_speed_limit import ASIF
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# class SystemParameters:
class agent:
    def __init__(self, mean_motion = 0.001027, mass_chaser = 140, max_available_thrust = 2, controller_sample_rate = 0.2, filter_sample_rate = 1,
                       f_goal_set = 2, total_plan_time = 1000, tau0 = 50, collision_dist = 500,
                       semi_minor_out_num = 5, semi_minor_in_num = np.sqrt(5/4)):

        ''' System parameters '''
        self.mean_motion = mean_motion # [rad/s]
        self.mass_chaser = mass_chaser # [kg]

        self.max_available_thrust = max_available_thrust # [N]

        self.controller_sample_rate = controller_sample_rate # [Hz]
        self.controller_sample_period = 1/controller_sample_rate # [s]

        self.filter_sample_rate = filter_sample_rate # [Hz]
        self.filter_sample_period = 1/filter_sample_rate # [s]

        ''' Controller setup '''
        self.f_goal_set = f_goal_set # 0 for origin, 1 for periodic line, 2 for ellipses
        self.f_collision_avoidance = True # activates or deactivates collision avoidance requirement

        self.total_plan_time = total_plan_time # time to goal [s]
        self.tau0 = tau0 # number steps in initial planning horizon
        self.collision_dist = collision_dist # minimum allowable distance from target [m]
        self.kappa_speed = 2.1*self.mean_motion # NOTE: must be greater than 2*mean_motion
        self.semiminor_out = semi_minor_out_num/self.mean_motion # semi-minor axis of outer ellipse bound - motivated by velocity constraint
        self.semiminor_in = semi_minor_in_num*self.collision_dist # semi-minor axis of inner ellipse bound - motivated by collision avoidance constraint

        # Set up (don't modify )
        self.zero_input = np.zeros([3,1])
        self.trajectory_initialized = False
        if self.f_goal_set == 0:
            self.f_collision_avoidance = False

        self.t = np.linspace(0,self.total_plan_time, self.tau0) # time vector
        self.dt_plan = self.t[1]-self.t[0] # time resolution of solver

        # self.xstar = 0 # optimal trajectory points
        self.ustar = 0 # optimal control points

        if self.f_goal_set == 0:
            print("\nDriving chaser to target! \n")
        elif self.f_goal_set == 1:
            print("\nDriving chaser to stationary trajectory! \n")
        elif self.f_goal_set == 2:
            print("\nDriving chaser to elliptical NMT! \n")


    def main(self, x0, t):
        """
        Computes trajectory and uses first input

        """

        # Try to find optimal trajectory
        try:
            self.calculate_trajectory(x0, t) # finds traj starting at x0
        except:
            try:
                self.tau0 = self.tau0 + 20
                self.calculate_trajectory(x0, t) # finds traj starting at x0
            except:
                print("\nFailed to find trajectory at t = "+str(t),"\n")

        u = self.ustar[:,0]

        return u.reshape(3,1)


    def calculate_trajectory(self, x0, t_elapsed):
        """
        Uses Gurobi to calculate optimal trajectory points (self.xstar) and
        control inputs (self.ustar)

        """

        # Options
        Nin = 6 # number of sides in inner-ellipse polygon approx
        Nout = 15 # number of sides in outer-ellipse polygon approx


        initial_state = x0.reshape(6)
        goal_state = np.zeros(6)
        n  = self.mean_motion
        mc = self.mass_chaser

        # Shorten the number of initial time steps (self.tau0) based on the amount of time elapsed
        tau = int(max(10, np.round(self.tau0 - t_elapsed/self.dt_plan) ) )
        print("time elapsed = ", t_elapsed )

        # Set Ranges
        smax = 15000 # arbitrary (included bounds to speed up solver)
        vmax = 10 # [m/s] max velocity
        Fmax = 2 # [N] max force


        # Initialize states
        sx = []
        sy = []
        sz = []
        vx = []
        vy = []
        vz = []
        Fx = []
        Fy = []
        Fz = []
        snorm = []
        sxabs = []
        syabs = []
        vnorm = []
        vxabs = []
        vyabs = []
        zeta = []

        m = gp.Model("QPTraj")

        # Define variables at each of the tau timesteps
        for t in range(tau) :
            sx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sx"+str(t) ))
            vx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vx"+str(t) ))
            Fx.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fx"+str(t) ))
            sy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sy"+str(t) ))
            vy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vy"+str(t) ))
            Fy.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fy"+str(t) ))
            sz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -smax, ub = smax, name="sz"+str(t) ))
            vz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -vmax, ub = vmax, name="vz"+str(t) ))
            Fz.append( m.addVar(vtype=GRB.CONTINUOUS, lb = -Fmax, ub = Fmax, name="Fz"+str(t) ))
            snorm.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="snorm"+str(t) ))
            sxabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="sxabs"+str(t) ))
            syabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = smax, name="syabs"+str(t) ))
            vnorm.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vnorm"+str(t) ))
            vxabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vxabs"+str(t) ))
            vyabs.append( m.addVar(vtype=GRB.CONTINUOUS, lb = 0, ub = vmax, name="vyabs"+str(t) ))

        for p in range(Nin):
            zeta.append( m.addVar(vtype=GRB.BINARY, name="zeta"+str(p)) )

        m.update()

        # Set Initial Conditions
        m.addConstr( sx[0] == initial_state[0] , "sx0")
        m.addConstr( sy[0] == initial_state[1] , "sy0")
        m.addConstr( sz[0] == initial_state[2] , "sz0")
        m.addConstr( vx[0] == initial_state[3] , "vx0")
        m.addConstr( vy[0] == initial_state[4] , "vy0")
        m.addConstr( vz[0] == initial_state[5] , "vz0")


        # Specify Terminal Set
        if self.f_goal_set == 0: # origin
            m.addConstr( sx[-1] == goal_state[0] , "sxf")
            m.addConstr( sy[-1] == goal_state[1] , "syf")
            m.addConstr( sz[-1] == goal_state[2] , "szf")
            m.addConstr( vx[-1] == goal_state[3] , "vxf")
            m.addConstr( vy[-1] == goal_state[4] , "vyf")
            m.addConstr( vz[-1] == goal_state[5] , "vzf")

        elif self.f_goal_set == 1: # stationary point or periodic line
            m.addConstr( sx[-1] == goal_state[0] , "sxf")
            m.addConstr( vx[-1] == goal_state[3] , "vxf")
            m.addConstr( vy[-1] == goal_state[4] , "vyf")

        elif self.f_goal_set == 2: # ellipse
            m.addConstr( vy[-1] + 2*n*sx[-1] == 0, "ellipse1" )
            m.addConstr( sy[-1] - (2/n)*vx[-1] == 0, "ellipse2" )


        # Set dynamic speed limit
        for t in range(tau):
            # Define the norms:
            m.addConstr( sxabs[t] == gp.abs_(sx[t]) )
            m.addConstr( syabs[t] == gp.abs_(sy[t]) )
            m.addConstr( snorm[t] == gp.max_(sxabs[t], syabs[t]), "snorm"+str(t) )
            m.addConstr( vxabs[t] == gp.abs_(vx[t]) )
            m.addConstr( vyabs[t] == gp.abs_(vy[t]) )
            m.addConstr( vnorm[t] == gp.max_(vxabs[t], vyabs[t]), "vnorm"+str(t) )

            # Speed limit constraint:
            m.addConstr( vnorm[t] <= self.kappa_speed*snorm[t] )


        # Collision Avoidance Constraint
        if self.f_collision_avoidance:
            for t in range(tau):
                m.addConstr( snorm[t] >= self.collision_dist)
            if initial_state[0]<self.collision_dist or initial_state[1]<self.collision_dist:
                print("\nERROR: Initial position is too close! Collision constraint violated!\n")

        # # Final point within [1km-5km] of target
        # m.addConstr( snorm[-1] <= 2000 )
        # m.addConstr( snorm[-1] >= 1000 )


        # Terminal constraint: inner polygonal approx on outer ellipse bound
        Nout = Nout+1
        aout = self.semiminor_out
        bout = self.semiminor_out*2
        theta = np.linspace(0, 2*np.pi, Nout)
        for j in range(0,Nout-1):
            x0 = aout*np.cos(theta[j])
            y0 = bout*np.sin(theta[j])
            x1 = aout*np.cos(theta[j+1])
            y1 = bout*np.sin(theta[j+1])
            alphax = y0-y1
            alphay = x1-x0
            gamma  = alphay*y1 + alphax*x1
            m.addConstr( alphax*sx[-1] + alphay*sy[-1] >= gamma , "OPA"+str(j) )

        # Terminal constraint: outer polygonal approx on inner ellipse bound
        if self.f_collision_avoidance :
            a_in = self.semiminor_in
            b_in = self.semiminor_in*2
            theta = np.linspace(0, 2*np.pi, Nin+1)
            big_M = 100000
            for j in range(0,Nin):
                x0 = a_in*np.cos(theta[j])
                y0 = b_in*np.sin(theta[j])
                c1 = (2*x0/(a_in**2))
                c2 = (2*y0/(b_in**2))
                cmag = np.sqrt(c1**2 + c2**2)
                c1 = c1/cmag
                c2 = c2/cmag
                m.addConstr( c1*sx[-1] + c2*sy[-1] - c1*x0 - c2*y0 - big_M*zeta[j]  >= - big_M, "IPA"+str(j) )
            m.addConstr( sum( zeta[p] for p in range(Nin) ) >= 0.5 )


        # Set Dynamics
        for t in range(tau-1) :
            # Dynamics
            m.addConstr( sx[t+1] == sx[t] + vx[t]*self.dt_plan , "Dsx_"+str(t))
            m.addConstr( sy[t+1] == sy[t] + vy[t]*self.dt_plan , "Dsy_"+str(t))
            m.addConstr( sz[t+1] == sz[t] + vz[t]*self.dt_plan , "Dsz_"+str(t))
            m.addConstr( vx[t+1] == vx[t] + sx[t]*3*n**2*self.dt_plan + vy[t]*2*n*self.dt_plan + Fx[t]*(1/mc)*self.dt_plan , "Dvx_"+str(t) )
            m.addConstr( vy[t+1] == vy[t] - vx[t]*2*n*self.dt_plan + Fy[t]*(1/mc)*self.dt_plan , "Dvy_"+str(t) )
            m.addConstr( vz[t+1] == vz[t] + (-n**2)*sz[t]*self.dt_plan     + Fz[t]*(1/mc)*self.dt_plan , "Dvz_"+str(t) )

        # Set Objective ( minimize: sum(Fx^2 + Fy^2) )
        obj = Fx[0]*Fx[0] + Fy[0]*Fy[0] + Fz[0]*Fz[0]
        for t in range(0, tau):
            obj = obj + Fx[t]*Fx[t] + Fy[t]*Fy[t] + Fz[t]*Fz[t]


        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam( 'OutputFlag', False )

        # Optimize and report on results
        m.optimize()


        # Save desired trajectory
        self.xstar = np.zeros([6, tau])
        self.ustar = np.zeros([3, tau])
        self.snorm = np.zeros([tau])
        self.vnorm = np.zeros([tau])

        for t in range(tau): # TODO: find quicker way to do this
            self.xstar[0,t] = m.getVarByName("sx"+str(t)).x
            self.xstar[1,t] = m.getVarByName("sy"+str(t)).x
            self.xstar[3,t] = m.getVarByName("vx"+str(t)).x
            self.xstar[4,t] = m.getVarByName("vy"+str(t)).x
            self.ustar[0,t] = m.getVarByName("Fx"+str(t)).x
            self.ustar[1,t] = m.getVarByName("Fy"+str(t)).x
            self.ustar[2,t] = m.getVarByName("Fz"+str(t)).x
            self.snorm[t]   = m.getVarByName("snorm"+str(t)).x
            self.vnorm[t]   = m.getVarByName("vnorm"+str(t)).x

        m.dispose()








    def simulate(self, collision_dist = 4000, T  = 2, dim_state = 6, dim_control = 3,
                       x = 1000, y = 0, z = 0, x_dot = 0.005, y_dot = 0, z_dot = 0):

        # Parameters
        Nsteps = T
        Fmax = self.max_available_thrust # [N]
        T_sample = self.controller_sample_period # [s]

        # Initial values

        x0 = np.array([[x],  # x
                       [y],  # y
                       [z],  # z
                       [x_dot],  # xdot
                       [y_dot],  # ydot
                       [z_dot]]) # zdot

        u0 = np.array([[0],  # Fx
                       [0],  # Fy
                       [0]]) # Fz

        ##############################################################################
        #                                Simulate                                    #
        ##############################################################################

        # Set up simulation
        t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon
        X = np.zeros([dim_state, Nsteps]) # state at each time
        U = np.zeros([dim_control, Nsteps]) # control at each time
        state_error = np.zeros([dim_state, Nsteps]) # State error at each time step

        dt = t[1]-t[0]

        X[:,0]=x0.reshape(dim_state)
        asif = ASIF() # Initialize ASIF class

        steps_per_sample = np.max([1, np.round(T_sample/dt)])
        effective_controller_period = steps_per_sample*dt
        print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
              " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")

        # Iterate over time horizon
        for i in range(1,Nsteps):
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Call Controller
            if (i-1)%steps_per_sample == 0:
                u = self.main(X[:,i-1], (i-1)*dt)

            # Filter Input
            if f_use_RTA_filter:
                u = asif.main(X[:,i-1], u)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # Saturate
            for j in range(3):
                if u[j,0] > Fmax:
                    u[j,0] = Fmax
                elif u[j,0] < -Fmax:
                    u[j,0] = -Fmax

            U[:,i] = u.reshape(dim_control) # record history of control inputs (optional)

            # Propagate
            xdot = ClohessyWiltshire.CW(X[:,i-1].reshape(dim_state,1) , u)*dt
            X[:,i] = X[:,i-1] + xdot.reshape(dim_state)


        return self.xstar, self.ustar, self.snorm






if __name__ == '__main__':
    # Flags
    f_plot_option = 2 # choose 0, 1, 2, or 3
    f_save_plot = True # saves a plot at end of simulation
    f_use_RTA_filter = False # filters the controllers input to ensure safety

    # Parameters
    # collision_dist = 4000 # [m]
    T  = 2 # total simulation time [s]
    Nsteps = T # number steps in simulation time horizon

    dim_state = 6;
    dim_control = 3;
    agent0 = agent(semi_minor_out_num = 5, semi_minor_in_num = np.sqrt(5/4))
    Fmax = agent0.max_available_thrust # [N]
    T_sample = agent0.controller_sample_period # [s]



    agent1 = agent(semi_minor_out_num = 9, semi_minor_in_num = np.sqrt(9/4))
    Fmax1 = agent1.max_available_thrust # [N]
    T_sample1 = agent1.controller_sample_period # [s]

    # Initial Values
    # x = 1000 # [m]
    # x_dot = 0.005
    # x0 = np.array([[x],  # x
    #                 [0],  # y
    #                 [0],  # z
    #                 [0],  # xdot
    #                 [0],  # ydot
    #                 [0]]) # zdot

    u0 = np.array([[0],  # Fx
                   [0],  # Fy
                   [0]]) # Fz

    # Setup filter paramters
    # P = np.identity(6)

    # Assign desired traj to actual
    X = []
    U = []
    X,U,S = agent0.simulate(x = 1000)
    t = np.linspace(0, T, X[0,:].__len__())



    Y = []
    V = []
    SIGMA = []
    Y,V,SIGMA = agent1.simulate(x = 1200)
#
#     X = []
#     U = []
#     X,U,S = agent1.simulate()
#     t = np.linspace(0, T, X[0,:].__len__())
#
#     # X1 = []
#     # U1 = []
#     # X1,U1,S1 = agent1.simulate(x = -1000)
#     # t = np.linspace(0, T, X1[0,:].__len__())
#
#
#     ##############################################################################
#     #                                Plotting                                    #
#     ##############################################################################

    try:
        f_speed_limit_const = asif.safety_constraint # 0 for none, 1 right cone, or 2 for other cone
    except:
        f_speed_limit_const = 0
    if not f_use_RTA_filter:
        f_speed_limit_const = 0

    # Style plot
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(10,10))
    axis_font = 9
    ax_label_font = 11
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)

    # Plot results
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.grid()
    # ax1.plot(X[0,:],X[1,:],X[2,:],'.', color='coral', markersize=marker_size, alpha=0.8)
    ax1.plot(X[0,:],X[1,:],X[2,:], color='blue', alpha=0.6)
    # ax1.plot(X[0,0:3],X[1,0:3],X[2,0:3], color='red', markersize=marker_size, alpha=0.6)
    ax1.plot(Y[0,:],Y[1,:], color='orange', linewidth=line_width, alpha=0.6)

    # ax1.plot(np.array([1, 1, -1, -1, 1])*collision_dist, np.array([-1, 1, 1, -1, -1])*collision_dist, 'r')
    K = 10000
    ax1.set_xlim( [-K, K] )
    ax1.set_ylim( [-K, K] )
    ax1.set_zlim( [-K, K] )
    # ax1.plot(X[0,0],X[1,0],X[2,0])
    # ax1.plot(0,0,0,'go', alpha=0.5)
    ax1.set_xlabel("x-position", fontsize=ax_label_font)
    ax1.set_ylabel("y-position", fontsize=ax_label_font)
    ax1.set_zlabel("z-position")
    plt.title("Trajectory", fontsize=ax_label_font)




    ##############################################################################
    ##############################################################################
    ##############################################################################



    x0 = X[:,-1]
    u0 = np.array([[0],  # Fx
                   [0],  # Fy
                   [0]]) # Fz

    y0 = Y[:,-1]
    v0 = np.array([[0],  # Fx
                   [0],  # Fy
                   [0]]) # Fz

    # Setup filter paramters

    # P = np.identity(6)
    T = 8000
    Nsteps = T

    ##############################################################################
    #                                Simulate                                    #
    ##############################################################################

    # Set up simulation
    t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon
    X_NMT = np.zeros([dim_state, Nsteps]) # state at each time
    U_NMT = np.zeros([dim_control, Nsteps]) # control at each time

    state_error = np.zeros([dim_state, Nsteps]) # State error at each time step

    dt = t[1]-t[0]

    X_NMT[:,0]=x0.reshape(dim_state)

    asif = ASIF() # Initialize ASIF class


    steps_per_sample = np.max([1, np.round(T_sample/dt)])
    effective_controller_period = steps_per_sample*dt
    print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
          " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")


    Y_NMT = np.zeros([dim_state, Nsteps]) # state at each time
    V_NMT = np.zeros([dim_control, Nsteps]) # control at each time

    state_error = np.zeros([dim_state, Nsteps]) # State error at each time step

    dt = t[1]-t[0]

    Y_NMT[:,0]=y0.reshape(dim_state)

    # Iterate over time horizon
    for i in range(1,Nsteps):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Call Controller
        if (i-1)%steps_per_sample == 0:
            u = u0

        U_NMT[:,i] = u.reshape(dim_control) # record history of control inputs (optional)
        V_NMT[:,i] = u.reshape(dim_control) # record history of control inputs (optional)

        # Propagate
        xdot = ClohessyWiltshire.CW(X_NMT[:,i-1].reshape(dim_state,1) , u)*dt
        X_NMT[:,i] = X_NMT[:,i-1] + xdot.reshape(dim_state)

        ydot = ClohessyWiltshire.CW(Y_NMT[:,i-1].reshape(dim_state,1) , u)*dt
        Y_NMT[:,i] = Y_NMT[:,i-1] + ydot.reshape(dim_state)


    # Plot results
    ax1.plot(X_NMT[0,:],X_NMT[1,:],X_NMT[2,:], color='green', alpha=0.7)
    ax1.plot(Y_NMT[0,:],Y_NMT[1,:],Y_NMT[2,:], color='coral', alpha=0.7)
    vmag = np.maximum(np.abs(X_NMT[3,:]), np.abs(X_NMT[4,:]))


    # Save and Show
    if f_save_plot:
        plt.savefig('trajectory_plot')
        plt.show()

    # End
    print("complete")

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # if __name__ == '__main__':
# #     # Flags
# #     f_plot_option = 2 # choose 0, 1, 2, or 3
# #     f_save_plot = True # saves a plot at end of simulation
# #     f_use_RTA_filter = False # filters the controllers input to ensure safety
# #
# #     # Parameters
# #     # collision_dist = 4000 # [m]
# #     T  = 2 # total simulation time [s]
# #     Nsteps = T # number steps in simulation time horizon
# #
# #     dim_state = 6;
# #     dim_control = 3;
# #     agent0 = agent(semi_minor_out_num = 5, semi_minor_in_num = np.sqrt(5/4))
# #     Fmax = agent0.max_available_thrust # [N]
# #     T_sample = agent0.controller_sample_period # [s]
# #
# #     # Initial Values
# #     x = 1000 # [m]
# #     x_dot = 0.005
# #     x0 = np.array([[x],  # x
# #                     [0],  # y
# #                     [0],  # z
# #                     [0],  # xdot
# #                     [0],  # ydot
# #                     [0]]) # zdot
# #
# #     u0 = np.array([[0],  # Fx
# #                    [0],  # Fy
# #                    [0]]) # Fz
# #
# #     # Setup filter paramters
# #     x_hat = x0
# #
# #     P = np.identity(6)
# #
# #     ##############################################################################
# #     #                                Simulate                                    #
# #     ##############################################################################
# #
# #     # Set up simulation
# #     t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon
# #     X = np.zeros([dim_state, Nsteps]) # state at each time
# #     U = np.zeros([dim_control, Nsteps]) # control at each time
# #     state_error = np.zeros([dim_state, Nsteps]) # State error at each time step
# #
# #     dt = t[1]-t[0]
# #
# #     X[:,0]=x0.reshape(dim_state)
# #     controller = agent() # Initialize Controller class
# #     asif = ASIF() # Initialize ASIF class
# #
# #     steps_per_sample = np.max([1, np.round(T_sample/dt)])
# #     effective_controller_period = steps_per_sample*dt
# #     print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
# #           " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")
# #
# #     # Iterate over time horizon
# #     for i in range(1,Nsteps):
# #         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #         # Call Controller
# #         if (i-1)%steps_per_sample == 0:
# #             u = agent0.main(X[:,i-1], (i-1)*dt)
# #
# #         # Filter Input
# #         if f_use_RTA_filter:
# #             u = asif.main(X[:,i-1], u)
# #         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #
# #         # Saturate
# #         for j in range(3):
# #             if u[j,0] > Fmax:
# #                 u[j,0] = Fmax
# #             elif u[j,0] < -Fmax:
# #                 u[j,0] = -Fmax
# #
# #         U[:,i] = u.reshape(dim_control) # record history of control inputs (optional)
# #
# #         # Propagate
# #         xdot = ClohessyWiltshire.CW(X[:,i-1].reshape(dim_state,1) , u)*dt
# #         X[:,i] = X[:,i-1] + xdot.reshape(dim_state)
# #
# #
# #     # Assign desired traj to actual
# #     X = []
# #     U = []
# #     X = agent0.xstar
# #     U = agent0.ustar
# #     s = agent0.snorm
# #     t = np.linspace(0, T, X[0,:].__len__())
# #
# #
# #     ##############################################################################
# #     #                                Plotting                                    #
# #     ##############################################################################
# #
# #     if f_plot_option == 0 :
# #         # Style plot
# #         marker_size = 5
# #         line_width = 2
# #         fig = plt.figure(figsize=(8,6))
# #         plt.grid()
# #         axis_font = 10
# #         ax_label_font = 10
# #         plt.xlabel("$x$", fontsize=ax_label_font)
# #         plt.ylabel("$y$", fontsize=ax_label_font)
# #         font = {'family' : 'normal',
# #                 'weight' : 'bold',
# #                 'size'   : axis_font}
# #         mpl.rc('font', **font)
# #
# #         # Plot results
# #         plt.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
# #         plt.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
# #         plt.plot(X[0,0],X[1,0],'kx')
# #         plt.xlim([-10000, 10000])
# #         plt.ylim([-10000, 10000])
# #
# #     elif f_plot_option == 1 :
# #         # Style plot
# #         marker_size = 1.5
# #         line_width = 1.25
# #         fig = plt.figure(figsize=(20,5))
# #         axis_font = 15
# #         ax_label_font = 15
# #         font = {'family' : 'normal',
# #                 'weight' : 'bold',
# #                 'size'   : axis_font}
# #         mpl.rc('font', **font)
# #
# #         # Plot results
# #         ax1 = fig.add_subplot(121)
# #         ax1.grid()
# #         ax1.plot(X[0,:],X[1,:],'.', color='coral', markersize=marker_size, alpha=0.8)
# #         ax1.plot(X[0,:],X[1,:], color='blue', linewidth=line_width, alpha=0.6)
# #         ax1.plot(X[0,0],X[1,0],'kx')
# #         ax1.set_xlabel("$x-position$", fontsize=ax_label_font)
# #         ax1.set_ylabel("$y-position$", fontsize=ax_label_font)
# #
# #         ax2 = fig.add_subplot(122)
# #         ax2.grid()
# #         ax2.plot(t, X[3,:],'.', color='r', markersize=marker_size, alpha=0.2)
# #         ax2.plot(t, X[4,:],'.', color='b', markersize=marker_size, alpha=0.2)
# #         ax2.plot(t, X[3,:], color='red', linewidth=line_width, alpha=0.6)
# #         ax2.plot(t, X[4,:], color='blue', linewidth=line_width, alpha=0.6)
# #         ax2.set_xlabel("time", fontsize=ax_label_font)
# #         ax2.set_ylabel("velocity", fontsize=ax_label_font)
# #
# #     elif f_plot_option == 2 :
# #         try:
# #             f_speed_limit_const = asif.safety_constraint # 0 for none, 1 right cone, or 2 for other cone
# #         except:
# #             f_speed_limit_const = 0
# #         if not f_use_RTA_filter:
# #             f_speed_limit_const = 0
# #
# #         # Style plot
# #         marker_size = 1.5
# #         line_width = 1.25
# #         fig = plt.figure(figsize=(10,10))
# #         axis_font = 9
# #         ax_label_font = 11
# #         font = {'family' : 'normal',
# #                 'weight' : 'bold',
# #                 'size'   : axis_font}
# #         mpl.rc('font', **font)
# #
# #         # Plot results
# #         ax1 = fig.add_subplot(111, projection='3d')
# #         ax1.grid()
# #         # ax1.plot(X[0,:],X[1,:],X[2,:],'.', color='coral', markersize=marker_size, alpha=0.8)
# #         ax1.plot(X[0,:],X[1,:],X[2,:], color='blue', alpha=0.6)
# #         ax1.plot(X[0,0:3],X[1,0:3],X[2,0:3], color='red', markersize=marker_size, alpha=0.6)
# #         # ax1.plot(np.array([1, 1, -1, -1, 1])*collision_dist, np.array([-1, 1, 1, -1, -1])*collision_dist, 'r')
# #         K = 10000
# #         ax1.set_xlim( [-K, K] )
# #         ax1.set_ylim( [-K, K] )
# #         ax1.set_zlim( [-K, K] )
# #         # ax1.plot(X[0,0],X[1,0],X[2,0])
# #         # ax1.plot(0,0,0,'go', alpha=0.5)
# #         ax1.set_xlabel("x-position", fontsize=ax_label_font)
# #         ax1.set_ylabel("y-position", fontsize=ax_label_font)
# #         ax1.set_zlabel("z-position")
# #         plt.title("Trajectory", fontsize=ax_label_font)
# #
# #         # ax2 = fig.add_subplot(224)
# #         # ax2.grid()
# #         # rmag = np.maximum(np.abs(X[0,:]), np.abs(X[1,:]))
# #         # vmag = np.maximum(np.abs(X[3,:]), np.abs(X[4,:]))
# #         # ax2.plot(rmag, vmag, color='r')
# #         # ax2.plot(np.linspace(0,9000,10), controller.kappa_speed*np.linspace(0,9000,10), '--')
# #         # ax2.set_xlim( [0, 5000] )
# #         # ax2.set_ylim( [0, 10]   )
# #         # ax2.set_xlabel("$\Vert r \Vert_\infty$", fontsize=ax_label_font)
# #         # ax2.set_ylabel("$\Vert v \Vert_\infty$", fontsize=ax_label_font)
# #         # plt.title("Velocity vs. Postion Norms", fontsize=ax_label_font)
# #         #
# #         #
# #         # ax3 = fig.add_subplot(223)
# #         # ax3.set_xlabel("time", fontsize=ax_label_font)
# #         # ax3.set_ylabel("velocity", fontsize=ax_label_font)
# #         # plt.title("Thrust vs. Time", fontsize=ax_label_font)
# #         # ax3.plot(t, U[0,:], '.', color='red', markersize=marker_size, alpha=0.8)
# #         # ax3.plot(t, U[1,:], '.', color='blue', markersize=marker_size, alpha=0.8)
# #         # ax3.plot(t, U[2,:], '.', color='green', markersize=marker_size, alpha=0.8)
# #         # ax3.plot(t, U[0,:], color='red', linewidth=line_width, alpha=0.2)
# #         # ax3.plot(t, U[1,:], color='blue', linewidth=line_width, alpha=0.2)
# #         # ax3.plot(t, U[2,:], color='green', linewidth=line_width, alpha=0.2)
# #         #
# #         # ax4 = fig.add_subplot(222, projection='3d')
# #         # plt.title("Position vs Speed")
# #         # ax4.grid()
# #         # # vmag = (X[3,:]**2 + X[4,:]**2)**(0.5)
# #         # ax4.plot( X[0,:], X[1,:], vmag, 'r' )
# #         # K2 = 10000
# #         # if f_speed_limit_const >= 0.5:
# #         #     x = np.arange(-K2, K2, 10)
# #         #     y = np.arange(-K2, K2, 10)
# #         #     x, y = np.meshgrid(x, y)
# #         #     R = np.sqrt(asif.K)*np.sqrt(x**2 + y**2)
# #         #     z = R
# #         # ax4.set_xlim( [-K, K] )
# #         # ax4.set_ylim( [-K, K] )
# #         # ax4.set_zlim( [0, 15] )
# #         # ax4.set_xlabel("x-position", fontsize=ax_label_font)
# #         # ax4.set_ylabel("y-position", fontsize=ax_label_font)
# #         # ax4.set_zlabel("velocity magnitude", fontsize=ax_label_font)
# #         #
# #         # # Plot the surface.
# #         # if f_speed_limit_const == 1:
# #         #     surf = ax4.plot_surface(x,y,z, cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.3)
# #         # elif f_speed_limit_const == 2:
# #         #     surf = ax4.plot_surface(x,y,np.sqrt(z), cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False, alpha=.3)
# #
# #
# #
# #     ##############################################################################
# #     ##############################################################################
# #     ##############################################################################
# #
# #
# #
# #     x0 = X[:,-1]
# #     u0 = np.array([[0],  # Fx
# #                    [0],  # Fy
# #                    [0]]) # Fz
# #
# #     # Setup filter paramters
# #     # x_hat = x0
# #
# #     P = np.identity(6)
# #     T = 8000
# #     Nsteps = T
# #
# #     ##############################################################################
# #     #                                Simulate                                    #
# #     ##############################################################################
# #
# #     # Set up simulation
# #     t = np.linspace(0, T, Nsteps) # evenly spaced time instances in simulation horizon
# #     X_NMT = np.zeros([dim_state, Nsteps]) # state at each time
# #     U_NMT = np.zeros([dim_control, Nsteps]) # control at each time
# #     # X_hat = np.zeros([dim_state, Nsteps]) # state estimate at each time step
# #
# #     state_error = np.zeros([dim_state, Nsteps]) # State error at each time step
# #     # X_meas = np.zeros([dim_state, Nsteps])
# #
# #     dt = t[1]-t[0]
# #
# #     X_NMT[:,0]=x0.reshape(dim_state)
# #
# #     controller = agent() # Initialize Controller class
# #     asif = ASIF() # Initialize ASIF class
# #
# #
# #     steps_per_sample = np.max([1, np.round(T_sample/dt)])
# #     effective_controller_period = steps_per_sample*dt
# #     print("\nSimulating with time resolution "+"{:.2f}".format(dt)+
# #           " s and controller period "+"{:.2f}".format(effective_controller_period)+" s \n")
# #
# #     # Iterate over time horizon
# #     for i in range(1,Nsteps):
# #         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #         # Call Controller
# #         if (i-1)%steps_per_sample == 0:
# #             u = u0
# #
# #         U_NMT[:,i] = u.reshape(dim_control) # record history of control inputs (optional)
# #
# #         # Propagate
# #         xdot = ClohessyWiltshire.CW(X_NMT[:,i-1].reshape(dim_state,1) , u)*dt
# #         X_NMT[:,i] = X_NMT[:,i-1] + xdot.reshape(dim_state)
# #
# #
# #     # Plot results
# #     # ax1.plot(X[0,:],X[1,:],X[2,:],'.', color='green', markersize=marker_size/2, alpha=0.2)
# #     ax1.plot(X_NMT[0,:],X_NMT[1,:],X_NMT[2,:], color='green', alpha=0.7)
# #     # vmag = (X[3,:]**2 + X[4,:]**2)**(0.5)
# #     vmag = np.maximum(np.abs(X_NMT[3,:]), np.abs(X_NMT[4,:]))
# #
# #     # ax4.plot(X[0,:], X[1,:], vmag, 'g' , linewidth=line_width/1.5, alpha=0.7)
# #     # rmag = np.maximum(np.abs(X[0,:]), np.abs(X[1,:]))
# #     # vmag = np.maximum(np.abs(X[3,:]), np.abs(X[4,:]))
# #     # ax2.plot(rmag, vmag, '--', color='g', alpha=0.5)
# #
# #
# #     # Save and Show
# #     if f_save_plot:
# #         plt.savefig('trajectory_plot')
# #         plt.show()
# #
# #     # End
# #     print("complete")
