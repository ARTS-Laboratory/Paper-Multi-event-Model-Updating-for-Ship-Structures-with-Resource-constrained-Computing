# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

fig, host = plt.subplots(figsize=(6.5,2.75))
fig.subplots_adjust(right=0.9)

par1 = host.twinx()
par2 = host.twiny()
                # Step i,                                Crack i
p1, = host.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ,[0, .0024, .0048, .0072, .0096, .012, .0144, .0168, .0192, .0216, .024],  "-o", color='gray', label="crack length")         
                # Step i                                 Roller i
p2, = par1.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [70, 70, 70, 70, 70 ,70, 71, 71, 71, 71, 71], "-d",color='gray', label="roller location (mm)")

p3, = par2.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, .00235, .0048, .00726, .01, .012, .0146, .0172, .0192, .022, .024],  "-*",color=cc[0], label="estimated crack length")
                
p4, = par1.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [70, 70, 70, 70, 70 , 70, 71, 71, 71, 71, 71], "--^",color=cc[1], label="estimated roller location ")
'''

p3, = par2.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, .0026, .0058, .0079, .008, .010, .016, .018, .0202, .022, .0235],  "-*",color=cc[0], label="estimated crack length")
                
p4, = par1.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [70, 70, 70, 70, 70 , 70, 71, 71, 71, 71, 71], "--^",color=cc[1], label="estimated roller location ")
                
'''

host.set_xlim(-.5, 10.5)
host.set_ylim(-.00175, .0255)
par1.set_ylim(69.93, 71.06)

#host.set_xlim(-.5, 10.5)
#host.set_ylim(-.0015, .02475)

#par2 = par1

host.set_xlabel("Steps")
host.set_ylabel("crack length (mm)")
par1.set_ylabel("roller location (mm)")


host.yaxis.label.set_color(cc[0])
par1.yaxis.label.set_color(cc[1])


tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=cc[0], **tkw)
par1.tick_params(axis='y', colors=cc[1], **tkw)

host.tick_params(axis='x', **tkw)

lines = [p3,p4]#p1, p2, 
host.legend(lines, [l.get_label() for l in lines], prop={'size': 9})
#host.legend(list, prop={'size': 12})

plt.tight_layout()
plt.savefig('crack_vs_impact_chart.pdf',dpi=500)


#mpl.pyplot.legend(list, prop={'size': 12})
















