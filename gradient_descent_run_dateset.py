# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

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
  

#%% define the paramaters

data_set = 'fatigue_and_impact_0.002_mm_step.pkl'

# list the nodes that heavly partisipate in the z axis
Z_modes = [2,4,5,7]#[2,4,5,7]   #[2,4,5,7]
# this lines up with the index as "mode" zero is displacement only


#%% load the ground truth data

trueDList = np.load('data/fatigue_and_impact_3_mm_step.pkl',allow_pickle=True)

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



#%% Gradient descent code


# pick the damage case to test at, 

beamState = 4
alpha = 30

print('reference crack length '+str(crackLengths[beamState]) +'; roller location '+str(rollerLocations[beamState]))


gridCrack, gridRoller = np.mgrid[0.012:0.018:25j, 0.70:0.72:25j]#[0.0012:0.0018:25j, 0.69:0.71:25j]#[0.018:0.024:25j, 0.69:0.71:25j]#[0.002:0.018:25j, 0.69:0.71:25j]
gridJ = np.zeros(gridCrack.shape)
gridJ1 = np.zeros(gridCrack.shape)
gridJ2 = np.zeros(gridCrack.shape)
gridF = np.zeros(gridCrack.shape)

modeData = []
for i in range(gridCrack.shape[0]):
    for ii in range(gridCrack.shape[1]):
        testCrack = gridCrack[i,ii]
        testRoller = gridRoller[i,ii]
        #print('testing crack length '+str(testCrack) +'; roller location '+str(testRoller))
        
        tryCounter = 0
        while tryCounter<5:
            try:
  
                # run the test case and orginize the code
                testD = fnx.FEA_crack(testCrack,testRoller)
                testDScaledModeShapes = fnx.scaleModeShapes(testD)
                
                testModeDispZ = testDScaledModeShapes['modeDispZ']
                testModeCoordX = testDScaledModeShapes['modeCoordX']
                testPartFactorZ = testDScaledModeShapes['partFactorZ']
                testFreq = testDScaledModeShapes['freq'][:,1]
                
                truePhi = trueModeDispZ[:,Z_modes]
                testPhi = testModeDispZ[:,Z_modes]
                # trueOmega = trueFreq[Z_modes]
                # testOmega = testFreq[Z_modes]
                # I think we have to offset the frequency index because the frequency numbers
                # start at mode 1 while the mode shapes start at 0. Also, python index from 
                # 0 so the frequency index does not line up with the mode index.
                Z_modes_freq = []
                for iii in range(len(Z_modes)):
                    Z_modes_freq.append(Z_modes[iii]-1)
                trueOmega = trueFreq[Z_modes_freq]
                testOmega = testFreq[Z_modes_freq]
                                
                #  compute the DMBA cost function
                alpha = 1
                modeData.append(testD)
                gridJ[i,ii], gridJ1[i,ii], gridJ2[i,ii] = fnx.DMBA(trueOmega,testOmega,truePhi,testPhi,alpha)
                #trueMmassConstants = trueD['effectiveMassZ'][:,1] 
                testMmassConstants = testD['effectiveMassZ'][:,1] # The -1 is because there 
                # is no effective mass value for the "0" mode shape. 
                gridF[i,ii] = fnx.flexibilityBasedApproach(trueModeDispZ,trueFreq,trueMmassConstants,testModeDispZ,testFreq,testMmassConstants,Z_modes)
                break
            
            except Exception as e:
                print(e)
                print('handling some unknown error, trying again, counter xxx '+ str(tryCounter))
                tryCounter =+1
                time.sleep(5)
                # this is a catch in case the last code did not return to the mane directory
                if os.path.basename(os.getcwd()) == 'abaqus_working_files':
                    os.chdir('..')
                
                # delete all the files in the directy. 
                files = glob.glob('abaqus_working_files/*')
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        print('could not delete '+str(f))
                
            
            if tryCounter ==4:
                print('handling some unknown error, returning NaN')
                modeData.append('NaN')
                gridJ[i,ii] = 'NaN'
        
        
#%% Save the data
    
saveData = {
    'gridCrack':gridCrack,
    'gridRoller':gridRoller,
    'gridJ':gridJ,
    'gridJ1':gridJ1,
    'gridJ2':gridJ2,
    'gridF':gridF,
    #'modeData':modeData, 
    'beamState':beamState,
    'Z_modes':Z_modes,
    'alpha':alpha,
    }   


pickle.dump(saveData,open('data/gradient_decent_'+str(len(Z_modes))+'_modes.pkl','wb'))


#%% Plot the results

fig = plt.figure()
plt.imshow(gridJ)
plt.savefig('plots/gradient_decent_alpha_'+str(alpha)+'.png', dpi=300)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('J')
surf = ax.plot_surface(gridCrack, gridRoller,gridJ, cmap=plt.cm.viridis,
                       linewidth=1, antialiased=False)
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('J1')
surf = ax.plot_surface(gridCrack, gridRoller,gridJ1, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J1')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('J2')
surf = ax.plot_surface(gridCrack, gridRoller,gridJ2, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('J2')




fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('F')
surf = ax.plot_surface(gridCrack, gridRoller,gridF, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlabel('crack length (m)')
ax.set_ylabel('roller location (m)')
ax.set_zlabel('F')















































































































































