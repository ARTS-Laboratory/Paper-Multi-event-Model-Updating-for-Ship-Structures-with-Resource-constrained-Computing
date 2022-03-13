# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
import fnx as fnx

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
plt.close('all')
  


#%% load the ground truth data

trueDList = np.load('data/fatigue_and_impact_0.002_mm_step.pkl',allow_pickle=True)

frequencies = []
crackLengths = []
rollerLocations = []

truePartFactorZ = []
for i in range(len(trueDList)):
    frequencies.append(trueDList[i]['frequencies'][:,1])
    crackLengths.append(trueDList[i]['crack_length'])
    rollerLocations.append(trueDList[i]['roller_location'])
    truePartFactorZ.append(trueDList[i]['participationFactorZ'])

crackLengths = np.asarray(crackLengths)
rollerLocations = np.asarray(rollerLocations)
frequencies = np.asarray(frequencies)



beamState = 4
trueD = trueDList[beamState]
trueMmassConstants = trueD['effectiveMassZ'][:,1] 

trueDScaledModeShapes = fnx.scaleModeShapes(trueD)
trueModeDispZ = trueDScaledModeShapes['modeDispZ']
trueModeCoordX = trueDScaledModeShapes['modeCoordX']
truePartFactorZ = trueDScaledModeShapes['partFactorZ']
trueFreq = trueDScaledModeShapes['freq'][:,1]


# # plot the mode shapes 
# plt.figure(figsize=(8,6)); iii=1
# for i in Z_modes:        
#     plt.subplot(3,2,iii)
#     plt.title('mode shape ' + str(i))
#     plt.grid(True)
#     plt.xlabel('beam length (m)')
#     plt.ylabel('mode shape')
#     plt.plot(trueModeCoordX,trueModeDispZ[:,i])
#     plt.tight_layout
#     iii += 1
# plt.tight_layout()


# plot the input
fig, ax1 = plt.subplots(figsize=(6.5,3))

color = cc[0]
ax1.set_xlabel('step')
ax1.set_ylabel('crack length (mm)', color=color)
ax1.plot(crackLengths*1000, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = cc[1]
ax2.set_ylabel('roller 1 displacement (mm)', color=color)  # we already handled the x-label with ax1
ax2.plot(rollerLocations*1000, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig('plots/inputs',dpi=300)

# plot the frequency components for the modes under consideration 
plt.figure(figsize=(6.5,6.55))
plt.subplot(211)
plt.plot(frequencies[:,0],'-',label = 'natural frequency')
plt.plot(frequencies[:,1],'--',label = '2nd frequency')
plt.plot(frequencies[:,2],'-.',label = '3rd frequency')
plt.plot(frequencies[:,3],':',label = '4th frequency')
plt.plot(frequencies[:,4],'-',label = '5th frequency')
plt.plot(frequencies[:,5],'--',label = '6th frequency')
plt.plot(frequencies[:,6],'-.',label = '7th frequency')
plt.plot(frequencies[:,7],':',label = '8th frequency')
plt.xlabel('step')
plt.ylabel('frequency (Hz)')
plt.grid(True)
# plt.legend(framealpha=1)
plt.tight_layout()
#plt.savefig('plots/frequencies',dpi=300)

def percent_change(x):
    x_0 = x[0]
    return(((x-x_0)/x_0)*100)

plt.subplot(212)
plt.plot(percent_change(frequencies[:,0]),'-',label = 'natural frequency')
plt.plot(percent_change(frequencies[:,1]),'--',label = '2nd frequency')
plt.plot(percent_change(frequencies[:,2]),'-.',label = '3rd frequency')
plt.plot(percent_change(frequencies[:,3]),':',label = '4th frequency')
plt.plot(percent_change(frequencies[:,4]),'-',label = '5th frequency')
plt.plot(percent_change(frequencies[:,5]),'--',label = '6th frequency')
plt.plot(percent_change(frequencies[:,6]),'-.',label = '7th frequency')
plt.plot(percent_change(frequencies[:,7]),':',label = '8th frequency')
plt.ylabel('%$\Delta$ frequency')
plt.xlabel('time (step)')
plt.grid(True)
plt.legend(framealpha=1)
plt.tight_layout()
#plt.savefig('plots/change_in_frequencies',dpi=300)



#%% plot the 3d surfance response plots

# loads the data for the direct model-based approach and flexibility matrix. 
number_modes = 4
surfaceD = np.load('data/gradient_decent_'+str(number_modes)+'_modes.pkl',allow_pickle=True)

for keys, vals in surfaceD.items():
    exec(keys + '=vals')


# adjust the alpha value as needed
alpha=30
gridJ = gridJ1 + alpha*gridJ2


# replace the high values with NaN for plotting
gridJ[gridJ>2000] = np.nan 
gridJ2[gridJ2>2000] = np.nan 


# plot J
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('direct model-based approach (J)')
data = gridJ
surf = ax.plot_surface(gridCrack, gridRoller,data, cmap=plt.cm.viridis,
                        linewidth=1, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J')
#ax.view_init(elev=46, azim=130)
#plt.savefig('plots/J_'+str(number_modes)+'_modes',dpi=300)

# plot J1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('J1')
data = gridJ1
surf = ax.plot_surface(gridCrack, gridRoller,data, cmap=plt.cm.viridis,
                        linewidth=1, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J1')
#plt.savefig('plots/J1_'+str(number_modes)+'_modes',dpi=300)

# plot J2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('J2')
data = gridJ2
surf = ax.plot_surface(gridCrack, gridRoller,data, cmap=plt.cm.viridis,
                        linewidth=1, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J2')
#plt.savefig('plots/J2_'+str(number_modes)+'_modes',dpi=300)

# plot F
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('flexibility matrix (F)')
data = gridF
surf = ax.plot_surface(gridCrack, gridRoller,data,cmap=plt.cm.viridis, 
                        linewidth=1, antialiased=False, vmin=np.nanmin(data), 
                        vmax=np.nanmax(data))
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')


plt.savefig('crack_vs_impact_chart',dpi=500)
#plt.savefig('plots/F_'+str(number_modes)+'_modes',dpi=500)
plt.show()

