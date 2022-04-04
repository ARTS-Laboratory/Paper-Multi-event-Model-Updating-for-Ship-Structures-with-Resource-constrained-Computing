# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:45:54 2021
@author: Jason Smith, Hung-Tien Huang
"""
#%% Imports
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
from sklearn import preprocessing as ps
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
# set default fonts and plot colors
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.close('all')

#%%
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
# set default fonts and plot colors
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%% Input Data
#C:\Users\Angel Nguyen\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\Surfaceplots\Random Cracks
DATA = np.load('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\Surfaceplots\Random Cracks\gradient_decent_4_modes_010_7.pkl',allow_pickle=True)
gridCrack = DATA['gridCrack']
gridRoller = DATA['gridRoller']
gridJ = DATA['gridJ']
gridJ1 = DATA['gridJ1']
gridJ2 = DATA['gridJ2']
gridF = DATA['gridF']

#%% Min of Flex Data
col_min_flex = gridF.min(axis=1)
#print(col_min_flex)
col_min_flex_index = np.argmin(gridF, axis=1)
#print(col_min_flex_index)
Min_Flex = col_min_flex  #.tolist()

#%% Min of Crack Length
Min_Crack = []
for i in range(len(col_min_flex_index)):
    Min_Crack.append(gridCrack[i, col_min_flex_index[i]])
    #print (Min_Crack)
Min_Crack_1 = np.array(Min_Crack)

#%% Min of Roller Location
Min_Roller = []
for i in range(len(col_min_flex_index)):
    Min_Roller.append(gridRoller[i, col_min_flex_index[i]])
    #print (Min_Roller)
Min_Roller_1 = np.array(Min_Roller)



# Plotting min values
#ax.scatter(Min_Crack,Min_Roller,Min_Flex,marker='o', cmap = 'Sequential')
#plt.plot(Min_Crack,Min_Roller,Min_Flex,color='r')
#ax.scatter(Min_Crack_1,Min_Roller_1,Min_Flex,marker='o',s=75)

#############################################################################################
# Plotting Lowest Flex and Crack data to check the shape
ax = plt.figure(figsize=(6.5, 2.5))
#ax.set_title('Scaled F using M.M fit_transform')
p1 = plt.plot(Min_Crack_1, Min_Flex, linewidth=1, marker='d', color=cc[1])
plt.xlabel('Crack Length')
plt.ylabel('F')
plt.title('Error shape')


#%%
def scale_data(grid_f_matrix: np.ndarray,
               feature_range: Tuple[int, int] = (0, 1),
               copy: bool = True):
    """Scale Grid F matrix
    Args:
        grid_f_matrix (np.ndarray): grid_f_matrix to be transformed
        feature_range (Tuple[int, int], optional): desired range of transformed data. Defaults to (0, 1).
        copy (bool, optional): set to false to perform inplace row normalization and avoid a copy. Defaults to True.
    Returns:
        transformed (np.ndarray): the transformed crack_matrix
        min_max_scalar (MinMaxScalar): the fitted MinMaxScalar, but np.expand_dims(a=grid_f_matrix.flatten(), axis=0).transpose() is requried prior to calling transform, and np.squeeze(transformed.transpose(), axis=0).reshape(input_shape) is required to transform back to original shape.
    """
    input_shape = grid_f_matrix.shape
    min_max_scaler: MinMaxScaler = MinMaxScaler(feature_range=feature_range,
                                                copy=copy)
    transformed: np.ndarray = min_max_scaler.fit_transform(
        np.expand_dims(a=grid_f_matrix.flatten(), axis=0).transpose())
    transformed = np.squeeze(transformed.transpose(), axis=0)
    transformed = transformed.reshape(input_shape)
    return transformed, min_max_scaler


#%% Feature Surface
# Min - Max Normalization
scaled_grid_crack, crack_min_max = scale_data(gridCrack)
scaled_grid_roller, roller_min_max = scale_data(gridRoller)
scaled_grid_f, f_min_max = scale_data(gridF)

#Plotting Min - Max Scalar fit_transform
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Scaled F using M.M fit_transform')
surf1 = ax.scatter(scaled_grid_crack,
                   scaled_grid_roller,
                   scaled_grid_f,
                   cmap=plt.cm.viridis,
                   linewidth=0,
                   antialiased=False)
ax.set_xlabel('crack length')
ax.set_ylabel('roller location')
ax.set_zlabel('F')
#plt.savefig(fname="../result/transformed.png", dpi=300)

#%% Particle Swarm for the scaled data
for x in range (2,21,2):
    swarm_size = x
    for y in range (5,41,5):
        iterations = y
        
        save_path = 'C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test'
        file_name = "Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt"

        completeName = os.path.join(save_path, file_name)
        #print(completeName)
            
        file1 = open(completeName, "a")
        #file1.write("file information")
        file1.close()
        
        
        for z in range(21):
            
            #swarm_size = 8
            dimensions = 3
            inertia = -5
            #iterations = 25
            GRIDF_data = []
            local_weight = 1.555 # weighted factor for the particles historical best
            global_weight = 1
            particle_locations = []
            data_set = []
            full_data_set = []
            max_velocity = .0615 # the highest velocity allowed for a particle
            step_size = 1  # step size for updating each particle, or how far a particle 
            
            #%Setup the PSO
            # loads the data for the direct model-based approach and flexibility matrix. 
            number_modes = 4
            #C:\Users\Angel Nguyen\Downloads\Surfaceplots\Between Crack Lengths\Roller_Locations_69_71\Every_Step\Step_0_0001_002
            #surfaceD = 
            # set the x,y,z location of the particle's (initial guess) scattered around the x=0,y=0
            All_gridF_data = scaled_grid_f# Z axis data
            #print (All_gridF_data[0][0])
            All_crack_data = scaled_grid_crack# X axis data
            #print (All_crack_data[0][0])
            All_roller_data = scaled_grid_roller# Y axis data
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
                particle_location = full_data_set[i+400]#full_data_set[i]   #random.choice(full_data_set)
                particle_locations.append(particle_location)  #Tuple
                i+=1
                
            #%% Comparing each of the particles to each other to determine which particle has the lowest values  
            def takeThird(elem):
                return elem[2] 
            
            particle_locations.sort(key=takeThird)
            lowest_coordinate = particle_locations[0]
            #print(lowest_coordinate)      
            #print(particle_locations) 
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
            
                            particle_velocity = np.random.uniform( .0615, -.0615, size = (swarm_size,dimensions))
                            
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
                particle_lowest_Z_value = np.min(particle_locations[:,2])
                particle_lowest_Z_value_index = np.argmin(particle_locations)
                #print("Lowest_Z", particle_lowest_Z_value)
                #print("Lowest_Z_Index", particle_lowest_Z_value_index)
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
            all_lowest_particles = np.array(all_lowest_particles)    
            flat_list = [item for sublist in all_lowest_particles for item in sublist] 
            particle_global_lowest_Z = np.min(all_lowest_particles[:,2])# Lowest Z value out of the lowest particle location after each location 
            b = np.where(np.isclose(all_lowest_particles,particle_global_lowest_Z))
            i_index = b[0]
            particle_global_lowest_Y = all_lowest_particles[i_index[0]][1]
            particle_global_lowest_X = all_lowest_particles[i_index[0]][0]
            particle_global_lowest_coordinate = [particle_global_lowest_X,particle_global_lowest_Y,particle_global_lowest_Z]
            
            A = np.where(np.isclose(scaled_grid_f,particle_global_lowest_Z))
            
            print("A index is",A)
            print("x",2)
            print("y",1)
            print("z",i_index)
            print(particle_global_lowest_X)
            print(particle_global_lowest_Y)
            print(particle_global_lowest_Z)
            print("LOWEST_COORD>",particle_global_lowest_coordinate )               
            #print (particle_locations)            

            Optimal_Location = [gridCrack[A],gridRoller[A],gridF[A]]
            print ("Optimal Location is", Optimal_Location[0][0],Optimal_Location[1][0],Optimal_Location[2][0])
            
            
            file1 = open(completeName, "a")
            file1.write(str(Optimal_Location[0][0]) + "," + str(Optimal_Location[1][0]) + ","  + str(Optimal_Location[2][0]))
            file1.write("\n")
            file1.close()
            



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Scaled F using M.M fit_transform')
surf1 = ax.scatter(scaled_grid_crack, scaled_grid_roller,scaled_grid_f,cmap=plt.cm.viridis, 
                        linewidth=.1, antialiased=False, vmin=np.nanmin(scaled_grid_f), 
                        vmax=np.nanmax(scaled_grid_f))

ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')

for i in range(len(iteration_location)-1):
    #ax.scatter(iteration_location[i][:,0],iteration_location[i][:,1],iteration_location[i][:,2],marker='x', cmap = 'Sequential')
    ax.scatter(iteration_location[i+1][:,0],iteration_location[i+1][:,1],iteration_location[i+1][:,2],linewidth =1,marker='x', cmap = 'Sequential')
ax.scatter(scaled_grid_crack[A],scaled_grid_roller[A],scaled_grid_f[A],linewidth =4,marker="*",)
#ax.scatter(iteration_location[0][:,0],iteration_location[0][:,1],iteration_location[0][:,2],marker='*', cmap = 'Sequential')
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')
plt.show()


#%% Plotting Flex, roller, crack
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('F')

surf0 = ax.scatter(gridCrack,
                   gridRoller,
                   gridF,
                   cmap=plt.cm.viridis,
                   linewidth=0,
                   antialiased=False)
ax.scatter(gridCrack[A],gridRoller[A],gridF[A],linewidth =4,marker="*",)
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')

Optimal_Location = [gridCrack[A],gridRoller[A],gridF[A]]
print ("Optimal Location is", Optimal_Location)
