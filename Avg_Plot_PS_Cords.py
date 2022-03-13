# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:09:15 2021

@author: jason
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
import statistics



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

#%% Summation Function
def summation(array, total):
    amount = 0
    for z in array:
        amount += z
    averge = amount/total 
    return averge

#%% Averaging and StDev F of the 20 runs of each combination 
for x in range (2,21,2):
    swarm_size = x
    for y in range (5,41,5):
        iterations = y
        with open ('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test' + '\\' + "Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt") as f:
            text = f.readlines()
        x = []
        y = [] 
        z = []
        for i in range(len(text)):
            result = text[i].split(",")
            x.append(float(result[0]))
            y.append(float(result[1]))
            z.append(float(result[2]))
        Average_x = summation(x, 21)          
        Average_y = summation(y, 21)
        Average_z = summation(z, 21)
        StDev = statistics.stdev(z)
        
        with open ('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test_Averages' + '\\' + "Averages_Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt", 'w+') as g:                          
            g.write(str(Average_x) + "," + str(Average_y) + "," + str(Average_z))   
        g.close()
        with open ('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test_StDev' + '\\' + "Averages_Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt", 'w+') as h:                          
            h.write(str(StDev))
        h.close()
#%% Opening and sorting Average and StDev data from .txt files 
Averages = []
StDevs = [] 

for x in range (2,21,2):
    swarm_size = x
    for y in range (5,41,5):
        iterations = y
        with open ('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test_Averages' + '\\' + "Averages_Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt") as f:
            text_average = f.readlines()     
            Averages.append(text_average)
        with open ('C:\\Users\jason\Documents\GitHub\multiEventModelUpdating\IWSHM_2021\Developing Frobenius norm search spaces\V1.1\PS_Test_StDev' + '\\' + "Averages_Swarm_" + str(swarm_size) + "Iteration" + str(iterations) + ".txt") as g:
            text_StDev = g.readlines()
            StDevs.append(text_StDev)
x_average_crack = []  
y_average_roller = [] 
z_average_F = []
for i in range(len(Averages)):
    result_average = Averages[i][0].split(",")
    x_average_crack.append(result_average[0])
    y_average_roller.append(result_average[1])
    z_average_F.append(result_average[2])

#x_average_crack.sort()
#y_average_roller.sort()
#z_average_F.sort()

x_average_crack = np.array(x_average_crack)
y_average_roller = np.array(y_average_roller)
z_average_F = np.array(z_average_F)

X_average_crack, Y_average_roller  = np.meshgrid(x_average_crack,y_average_roller)


Z_average_F = np.meshgrid(z_average_F,z_average_F)
ZZ_average_F = Z_average_F[1]


XX_average_crack = X_average_crack.astype(np.float)
YY_average_roller = Y_average_roller.astype(np.float)
ZZZ_average_F= ZZ_average_F.astype(np.float)


#%% Plotting 3-D Surface and Scatter Average Crack, Average Roller, Average F

fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.set_title('Averaged Coordinates')


'''
X_data = X_Coordinates_array[:,[0,1]]
Y_data = Y_Coordinates_array[[0,1],:]
Y_data = np.swapaxes(Y_data,0,1)
Z_data = ZZZ_average_F[:,[0,1]]
'''




X_data = XX_average_crack[0,:]
X_data = np.swapaxes(X_data,0,0)
Y_data = YY_average_roller[:,0]
Z_data = ZZZ_average_F[:,0]


'''
surf1 = ax.plot_trisurf(X_data, Y_data,Z_data,cmap=plt.cm.viridis, 
                        linewidth=0, antialiased=False)#, vmin=np.nanmin(ZZZ_average_F), vmax=np.nanmax(ZZZ_average_F))

'''
for i in range(len(z_average_F)):
    #ax.scatter(iteration_location[i][:,0],iteration_location[i][:,1],iteration_location[i][:,2],marker='x', cmap = 'Sequential')
    ax.scatter(float(x_average_crack[i]),float(y_average_roller[i]),float(z_average_F[i]),marker='x', linewidth=.5,antialiased=False)
    ax.text(float(x_average_crack[i])+0.00001,float(y_average_roller[i])+0.000001,float(z_average_F[i])+0.000001,str(i),fontsize=6)

ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')    
     
#%% Automatically creating and seperating # of Particles, # Iterations, Average F data for 3D plotting
xy_Coordinates = []

for i in range (2,21,2):
    Number_of_Particles = i
    for j in range (5,41,5):
        Number_of_iterations = j
        #for k in range(len(z)):
        xy_Coordinate = [Number_of_Particles,Number_of_iterations]#,float(z_average_F[0])]   
        xy_Coordinates.append(xy_Coordinate)
        
x_Coordinates = []   
y_Coordinates = [] 
Completed_Cords = []
        
for x in range(len(Averages)):
   x_Coordinates.append(xy_Coordinates[x][0]) 
   y_Coordinates.append(xy_Coordinates[x][1])
   Completed_Cord = [x_Coordinates[x],y_Coordinates[x],float(z_average_F[x])]
   Completed_Cords.append(Completed_Cord)

Completed_Cords = np.array(Completed_Cords)

#%% Plotting 3-D Scatter # of Particles, # Iterations, Average F
fig2 = plt.figure()
ay = fig2.gca(projection='3d')
ay.set_title('Averaged F')     
        
for i in range(len(Averages)):
    ay.scatter(x_Coordinates[i],y_Coordinates[i],float(z_average_F[i]),marker='x', linewidth=.5,antialiased=False)
    ay.text(float(x_Coordinates[i]) + 0.000001, float(y_Coordinates[i]) + 0.0000001, float(z_average_F[i]) + 0.0000001, str(i), fontsize=6)       
          
ay.set_xlabel('# of particles')
ay.set_ylabel('# of iterations')
ay.set_zlabel('Average F')  

#%% Plotting 3-D Surface Plot # of Particles, # Iterations, Average F
fig3 = plt.figure()
az = fig3.gca(projection='3d')
az.set_title('Averaged F')  

# x y and z from list to array for surface plot
x_Coordinates_array = np.array(x_Coordinates)
x_Coordinates_array = x_Coordinates_array.astype(float)
y_Coordinates_array = np.array(y_Coordinates)
y_Coordinates_array = y_Coordinates_array .astype(float)
z_average_F_array = np.array(z_average_F)

Y_Coordinates_array ,X_Coordinates_array = np.meshgrid(y_Coordinates_array , x_Coordinates_array)


surf2 = az.plot_trisurf(x_Coordinates_array, y_Coordinates_array ,ZZZ_average_F[:,0],cmap=plt.cm.viridis, 
                        linewidth=0, antialiased=False)#, vmin=np.nanmin(ZZZ_average_F), vmax=np.nanmax(ZZZ_average_F))


az.set_xlabel('# of particles')
az.set_ylabel('# of iterations')
#az.set_zlabel('Average F') 
az.view_init(15, 70)
plt.show() 
#plt.savefig('crack_vs_impact_chart',dpi=500)
#%% Automatically creating and seperating Standard Dev. F  data for 3D plotting
    
fig4 = plt.figure()
aZ = fig4.gca(projection='3d')
aZ.set_title('StDev F')     
        
for i in range(len(Averages)):
    aZ.scatter(x_Coordinates[i],y_Coordinates[i],float(StDevs[i][0]),marker='x', linewidth=.5,antialiased=False)
    aZ.text(float(x_Coordinates[i]) + 0.000000000001, float(y_Coordinates[i]) + 0.0000000000001, float(z_average_F[i]) + 0.0000000000001, str(i), fontsize=6)       
        
  
        
aZ.set_xlabel('# of particles')
aZ.set_ylabel('# of iterations')
aZ.set_zlabel('average F')    
    
#%% Plotting 3-D Surface Plot StDev # of Particles, # Iterations, Average F       
fig4 = plt.figure()
aZz = fig4.gca(projection='3d')
aZz.set_title('averaged' + '/n' + 'standard deviation')  

StDevs = np.array(StDevs)
StDevs = StDevs.astype(np.float)

StDevs = np.meshgrid(StDevs,StDevs)


Z_average_F = np.meshgrid(z_average_F,z_average_F)



surf2 = aZz.plot_trisurf(x_Coordinates_array, y_Coordinates_array ,StDevs[1][:,0],cmap=plt.cm.viridis, 
                        linewidth=1, antialiased=False)#, vmin=np.nanmin(ZZZ_average_F), vmax=np.nanmax(ZZZ_average_F))

aZz.set_xlabel('# of particles')
aZz.set_ylabel('# of iterations')
aZz.set_zlabel('standard' + '\n' + 'deviation') 
aZz.view_init(15, 70)
plt.show() 
plt.savefig('crack_vs_impact_chart',dpi=500)



 