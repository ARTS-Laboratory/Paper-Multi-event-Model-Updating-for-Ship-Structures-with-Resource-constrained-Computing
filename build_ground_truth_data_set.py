# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from numpy import fft
import subprocess as sp
import pickle as pickle
import os as os
import json as json
import glob as glob
import time as time
import fnx as fnx

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
plt.close('all')


#%% Run the data set
# [0.001,0.002D,0.003D,0.004D,0.005,0.006D,0.007,0.008,0.009,0.010]:0.011D
for ii in [0.002]:
    plt.close('all')
    num_steps = 10
    crack_length_start = 0.012
    crack_length_end = 0.0188
    crack_lengths = np.linspace(crack_length_start,crack_length_end,num_steps)
    roller_event_step = int(num_steps/2)
    roller_step = ii#0.001
    roller_location_start = 0.71
    roller_location_end = roller_location_start+roller_step
    roller_location_event_step = int(num_steps/2)
    roller_locations = np.zeros(num_steps)
    for i in range(num_steps):
        if i <  roller_location_event_step:
            roller_locations[i] = roller_location_start
        else:
            roller_locations[i] = roller_location_end
        
    #%% Run the Abaqus code
    data = []
    for i in range(num_steps):    
        crack_length = crack_lengths[i]
        roller_location = roller_locations[i]
        dataOutput = fnx.FEA_crack(crack_length,roller_location,deleteFiles=True,ignoreSimdir=True)            
        data.append(dataOutput)
   
    
    #%% Save the data to a pickle
    pickle.dump(data,open('data/fatigue_and_impact_'+'3'+'_mm_step.pkl','wb'))#str(roller_step)
    
    f = []
    for i in range(num_steps):
        f.append(data[i]['frequencies'])
    
    #%% Plot the results for this roller movement   
    frequencies = np.asarray(f)
        
    plt.figure(figsize=(6.5,5))
    plt.plot(frequencies[:,0,1],'-',label = 'natural frequency')
    plt.plot(frequencies[:,1,1],'--',label = '2nd frequency')
    plt.plot(frequencies[:,2,1],'-.',label = '3rd frequency')
    plt.plot(frequencies[:,3,1],':',label = '4th frequency')
    plt.plot(frequencies[:,4,1],'-',label = '5th frequency')
    plt.plot(frequencies[:,5,1],'--',label = '6th frequency')
    plt.plot(frequencies[:,6,1],'-.',label = '7th frequency')
    plt.plot(frequencies[:,7,1],':',label = '8th frequency')
    plt.xlabel('time (step)')
    plt.ylabel('frequency (Hz)')
    plt.grid(True)
    plt.legend(framealpha=1)
    plt.tight_layout()
    plt.savefig('plots/frequencies',dpi=300)
    
    
    fig, ax1 = plt.subplots(figsize=(6.5,3))
    color = cc[0]
    ax1.set_xlabel('time (step)')
    ax1.set_ylabel('crack length (mm)', color=color)
    ax1.plot(crack_lengths*1000, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = cc[1]
    ax2.set_ylabel('roller 1 displacement (mm)', color=color)  # we already handled the x-label with ax1
    ax2.plot(roller_locations*1000, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.savefig('plots/inputs',dpi=300)
    
    
    
    
    














