# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:17:42 2022

@author: jason
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import time as time
import sklearn as sklearn
from sklearn import linear_model
from sklearn import pipeline
from sklearn import datasets
from sklearn import multiclass
from sklearn import neural_network
import json as json
import pickle as pickle

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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
     
        

plt.figure(figsize=(6.5,5))    
plt.plot([0, 1, 2, 3, 4] ,[0, .045, .125, .045, 0],  "-d", color='blue', label="Mode 1")  
plt.xlabel('Acc. Location')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(framealpha=1)
plt.tight_layout()


plt.figure(figsize=(6.5,5))  
plt.plot([0, 1, 2, 3, 4], [0, .05, 0, .045, 0], "-d",color='blue', label="Mode 2")       
plt.xlabel('Acc. Location')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(framealpha=1)
plt.tight_layout()


plt.figure(figsize=(6.5,5))
plt.plot([0, 1, 2, 3, 4, 5, 6], [0, .032, 0, -.017, 0 ,.024, 0], "-d",color='blue', label="Mode 3")    
plt.xlabel('Acc. Location')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(framealpha=1)
plt.tight_layout()






















