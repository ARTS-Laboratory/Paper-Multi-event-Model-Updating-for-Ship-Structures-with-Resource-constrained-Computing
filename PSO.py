# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:17:52 2020

@author: Angel Nguyen
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
import numpy as np
import pylab as pl
from numpy import fft
import subprocess as subprocess
import scipy as sp
from scipy import interpolate
import pickle as pickle
import os as os
import glob
import time as time
import json as json
from scipy.spatial import KDTree
cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
plt.close('all')
# Set the parameters for the particle swarm optimization.
swarm_size = 10
dimensions = 3
inertia = 10
iterations = 20
GRIDF_data = []
local_weight = .705 # weighted factor for the particles historical best
global_weight = .145
particle_locations = []
data_set = []
full_data_set = []
max_velocity = .002 # the highest velocity allowed for a particle
step_size = 1  # step size for updating each particle, or how far a particle   

#%% Setup the PSO
# loads the data for the direct model-based approach and flexibility matrix. 
number_modes = 4
#C:\Users\Angel Nguyen\Downloads\Surfaceplots\Between Crack Lengths\Roller_Locations_69_71\Every_Step\Step_0_0001_002
surfaceD = np.load('C:\\Users\jason\Documents\All Suface Plots\gradient_decent_4_modes_010_7.pkl',allow_pickle=True)
# set the x,y,z location of the particle's (initial guess) scattered around the x=0,y=0
All_gridF_data = surfaceD['gridF']# Z axis data
#print (All_gridF_data[0][0])
All_crack_data = surfaceD['gridCrack']# X axis data
#print (All_crack_data[0][0])
All_roller_data = surfaceD['gridRoller']# Y axis data
#print (All_roller_data[0][1])


# Define the function that will be optimized. 
def Function(x0,y0,z0,x1,y1,z1,t):
    x = (x1-x0)*t + x0
    y = (y1-y0)*t + y0
    z = (z1-z0)*t + z0
    return(x,y,z)

for i in range(25):
    for j in range(25):
        data_set.append(All_crack_data[i][j])# X
        data_set.append(All_roller_data[i][j])# Y
        data_set.append(All_gridF_data[i][j])# Z
        full_data_set.append(data_set)
        data_set = []
neigbors = np.array(full_data_set)
kdtree=KDTree(neigbors)

#%% Re-starting particle locations
i = 0
while i < swarm_size:
    particle_location = random.choice(full_data_set)#full_data_set[i]   #random.choice(full_data_set)
    particle_locations.append(particle_location)  #Tuple
    i+=1
    
#%% Comparing each of the particles to each other to determine which particle has the lowest values  
def takeThird(elem):
    return elem[2] 

particle_locations.sort(key=takeThird)
lowest_coordinate = particle_locations[0]
print(lowest_coordinate)      
print(particle_locations) 
particle_locations = np.array(particle_locations)

#%%Setting up PSO
# create empty lists that are updated during the processes and used only for plotting the final results
best_value = []                  # for the best fitting value
best_locaion = []                # for the location of the best fitting value 
iteration_value_best = []        # for the best fitting value of the iteration
iteration_locaion_best = []      # for the location of the best fitting value of the iteration
iteration_value = []             # for the values of the iteration
iteration_locaion = [] 
particles = []
all_lowest_particles = []
iteration_location = [] 
#%% Moving particle 
for iteration_i in range(iterations): # for each iteration
    
    for particle_i in range(swarm_size): # for each particle
        
        for dimension_i in range(dimensions): # for each dimension
            #print(particle_locations)
            
            if iteration_i < iterations +1:
                
                # generate 2 random numbers between 0 and 1
                u = np.random.uniform( -100, 100, size = (2))  
                t = np.random.uniform( 0, 1, size = (1))

                particle_velocity = np.random.uniform( 0, .001, size = (swarm_size,dimensions))
                
                # solve the function for the particle's locations and save as their local best
                particle_best_value = Function(lowest_coordinate[0],lowest_coordinate[1],lowest_coordinate[2],particle_locations[:,0] ,particle_locations[:,1] ,particle_locations[:,2] ,t)#NEEDS TO BE LOWEST CURRENT PARTICLE
                particle_best_location = np.copy(particle_locations)

                # find the global best location of the initial guess and update the global best location
                global_best_value = np.min(particle_best_value)#XX
                global_best_location = np.array(lowest_coordinate)
                
                # calculate the error between the particle's best location and its current location
                error_particle_best = particle_best_location[particle_i,dimension_i] - \
                    particle_locations[particle_i,dimension_i]
                # calculate the error between the global best location and the particle's current location
                error_global_best = global_best_location[dimension_i] - \
                    particle_locations[particle_i,dimension_i]
                
                v_new = inertia*particle_velocity[particle_i,dimension_i] + \
                                local_weight*u[0]*error_particle_best + \
                                global_weight*u[1]*error_global_best
                
                # bound a particle's velocity to the maximum value set above       
                if v_new < -max_velocity:
                    v_new = -max_velocity
                elif v_new > max_velocity:
                    v_new = max_velocity
                
                # update the particle location
                particle_locations[particle_i,dimension_i] = (particle_locations[particle_i,dimension_i] + \
                                v_new*step_size)
                
                # nearest neighboor 
                for i in range(swarm_size):
                    sample = [particle_locations[i][0],particle_locations[i][1],particle_locations[i][2]]
                    sample = np.array(sample)
                    dist,points=kdtree.query(sample,1) 
                    particle_locations[i][0] = full_data_set[points][0]
                    particle_locations[i][1] = full_data_set[points][1]
                    particle_locations[i][2]= full_data_set[points][2]
                    
                # update the particle velocity
                particle_velocity[particle_i,dimension_i] = v_new
                    
    #finding lowest particle in the swarm every iteration 
    particle_lowest_Z_value = np.min(particle_locations)
    particle_lowest_Z_value_index = np.argmin(particle_locations)
    print("Lowest_Z", particle_lowest_Z_value)
    print("Lowest_Z_Index", particle_lowest_Z_value_index)
    lowest_Y_value_index = particle_lowest_Z_value_index - 1
    lowest_X_value_index = particle_lowest_Z_value_index - 2
    
    #Resize prticle locations so it can be looped through
    a = particle_locations.reshape(1,swarm_size*3)
    
    particle_lowest_Y_value = a[0][lowest_Y_value_index]
    particle_lowest_X_value = a[0][lowest_X_value_index]
    
    particle_lowest_coordinate = [particle_lowest_X_value,particle_lowest_Y_value,particle_lowest_Z_value]
    all_lowest_particles.append(particle_lowest_coordinate)
    
    print("Lowest_cord", particle_lowest_coordinate)
                
                
                
                
    iteration_location.append(particle_locations.copy()) 
    
flat_list = [item for sublist in all_lowest_particles for item in sublist] 
particle_global_lowest_Z = min(flat_list)# Lowest Z value out of the lowest particle location after each location 
b = flat_list.index(particle_global_lowest_Z)
particle_global_lowest_Y = flat_list[b - 1]
particle_global_lowest_X = flat_list[b - 2]
particle_global_lowest_coordinate = [particle_global_lowest_X,particle_global_lowest_Y,particle_global_lowest_Z]
print(b)
print(particle_global_lowest_Z)
print("LOWEST_COORD>",particle_global_lowest_coordinate )               
#print (particle_locations)            
'''
with open("Lowest_Cord.txt", "a") as f:
    for i in range(3):
        f.write(str(particle_global_lowest_coordinate[i]))
    f.write("\n")
    f.close()
'''
#%% Plotting
for keys, vals in surfaceD.items():
    exec(keys + '=vals')
    
# plot F
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Flexibility matrix (F)')
data = gridF

ax.scatter([0.01],[0.7],[0.0],marker='*', color = 'magenta',linewidth= 1)
surf1 = ax.plot_surface(gridCrack, gridRoller,data,cmap=plt.cm.viridis, 
                        linewidth=.001, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))


'''
surf1 = ax.scatter(gridCrack, gridRoller,data,cmap=plt.cm.viridis, 
                        linewidth=1, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))


ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')


for i in range(len(iteration_location)-1):
    #ax.scatter(iteration_location[i][:,0],iteration_location[i][:,1],iteration_location[i][:,2],marker='x', cmap = 'Sequential')
    ax.scatter(iteration_location[i+1][:,0],iteration_location[i+1][:,1],iteration_location[i+1][:,2],marker='x', cmap = 'Sequential')
#ax.scatter(iteration_location[0][:,0],iteration_location[0][:,1],iteration_location[0][:,2],marker='*', cmap = 'Sequential')
'''

ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F', fontweight='bold')
ax.view_init(35, -45)

plt.show()

print(all_lowest_particles[0][0])
plt.savefig('crack_vs_impact_chart',dpi=500)





