# -*- coding: utf-8 -*-
"""
Austin Downey
Jason Smith

Functions for running abaqus code
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from numpy import fft
import subprocess as subprocess
import scipy as sp
from scipy import interpolate
import pickle as pickle
import os as os
import time as time
import json as json
import glob as glob



def FEA_crack(crack_length,roller_location,deleteFiles=True,numTrys=5,ignoreSimdir=False):
    """
    This function runs the FEA-crack code and deals with unknown errors in the
    process by repeating up to the numTrys number. numTrys is default set to 5.
    
    """
    
    tryCounter = 1
    while tryCounter<numTrys:
        try:
            data_output = FEA_crack_run(crack_length,roller_location,deleteFiles,ignoreSimdir)
            break #if the complete successfully, this breaks out of the while 
            # loop. This also skips the last if statement.
        
        except:
            print('handling some unknown error, trying again, counter '+ str(tryCounter))
            tryCounter += 1
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
        
        # if the counter is at the end, this if statment returns a NaN value
        # for the data output dictionary. 
        if tryCounter == numTrys:
            print('could not handle an unknown error, counter '+ str(tryCounter) + ', returning NaN')
            data_output = 'NaN'
    
    return(data_output)


def FEA_crack_run(crack_length,roller_location,deleteFiles,ignoreSimdir):

    # catch in case the last code crashed and did not return to the main directory
    if os.path.basename(os.getcwd()) == 'abaqus_working_files':
        os.chdir('..')
    
    # create abaqus working directry if it does not exist
    if not os.path.exists('abaqus_working_files'):
        os.makedirs('abaqus_working_files')
    
    # change to the abaqus working file directory
    os.chdir('abaqus_working_files')

    # remove the old files to cause an error if this code does not write its own

    # takes a file name as a string and deletes its.
    def removeDataFiles(fName,sleepTime=5): 
        try:
            os.remove(fName)
        except FileNotFoundError:
            pass
        except PermissionError:
            print("retrying to delete "+fName)
            time.sleep(5)
            os.remove('data_in.json')   
            print(fName +" could not be removed")        
    
    # remove the data input and output files
    removeDataFiles('data_out.json')
    removeDataFiles('data_in.json')
    
    # the mesh size has to be outside to allow it to remesh if needed
    mesh_size= 0.0065 # 0.00564 # was 0.038

    # set a flag to make sure the FEA completed, this will keep running till FEAComplete=True
    FEAComplete=False    
    FEATry = 0
    while FEATry < 5: #Complete == False:
        
        FEATry = FEATry+1
        
        # define static paramaters
        crack_width = 0.001
        mesh_deviationFactor=0.1
        mesh_minSizeFactor=0.1
            
        # build the abaqus input file and write it to data_in.json
        data = {'crack_length':crack_length,
                'roller_location':roller_location,
                'mesh_deviationFactor':mesh_deviationFactor,
                'mesh_minSizeFactor':mesh_minSizeFactor,
                'mesh_size':mesh_size,
                'crack_width':crack_width,
                }
        
        with open('data_in.json', 'w') as f:
            json.dump(data, f)
        f.close()   # force close as anaconda compiler sometimes has issues closing
                
        # Run the Abaqus code
        print('crack length '+str(crack_length)+'; roller location '+str(roller_location))
        p = subprocess.call(r'C:\SIMULIA\Commands\abaqus cae -noGUI ../model_crack.py', shell=True)
                
        # check if an output file was created.  If not, wait 5 seconds and move 
        # to the next iteration in the while loop.
        try:
            with open('data_out.json') as f:
                odb_data = json.load(f)
            f.close()
        except FileNotFoundError:
            print("""XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                  no output file, code repeating
                  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX """)
            time.sleep(5)
            continue    # this should move to the next iteration in the while loop
            
        # If there is an issue with an element being buit between 2 nodes (no area)
        # the code returns without any frequecy data. If this is the case, remesh
        # the FEA and skip to the next iteration.
        freqData = np.asarray(odb_data['freqData'])
        if freqData.shape == ():
            print("""XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                  Frequency step error, adjusting mesh, code repeating
                  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX """)
            mesh_size = mesh_size + np.random.randn(1)[0]/10000
            continue
             
        # post process the displacement shapes into a mode shape that acts 
        # along the center of the beam.
        numFrames = len(odb_data['frame'])
        centerModeDisplacementsUnordered = []
        for i_frame in range(numFrames):   
            frame = odb_data['frame'][i_frame]          
            centerNodeLabels = odb_data['centerNodeLabels']     # node labels on the center line
            modeDisplacement = frame['modeDisplacement']    # displacements of all nodes in the model
            modeNodes = frame['modeNodes']  # The label for the nodes, this aligns with the displacements in modeDisplacement
            centerNodeCoordinates = np.asarray(odb_data['centerNodeCoordinates']) # the X, Y, Z foordinates for the nodes in their orginal position
            
            dd = []
            for i in range(len(centerNodeLabels)):             
                # find the index of the centerNodeLabels in modeNodes
                index = np.where(np.in1d(modeNodes, centerNodeLabels[i]))[0][0]   # the [0][0] are needed to get the index of the data point           
                dd.append(modeDisplacement[index])
            centerModeDisplacementsUnordered.append(dd)
        
        # change the list to an array and convert to X, Y, and X matricies to help make it 
        # easier to manage. 
        centerModeDisplacementsUnordered = np.asarray(centerModeDisplacementsUnordered)
        centerLineNodeDisplacementsXUnordered = centerModeDisplacementsUnordered[:,:,0]
        centerLineNodeDisplacementsYUnordered = centerModeDisplacementsUnordered[:,:,1]
        centerLineNodeDisplacementsZUnordered = centerModeDisplacementsUnordered[:,:,2]
        
        # Sort the node coordinats and retrive the index for their orginisation 
        index_order = centerNodeCoordinates[:,0].argsort()
        
        # rearrange the displacements (mode shapes) so the matriceis are in order of their 
        # respective place on the centerline
        centerLineNodeDisplacementsX = np.zeros(centerLineNodeDisplacementsXUnordered.shape)
        centerLineNodeDisplacementsY = np.zeros(centerLineNodeDisplacementsYUnordered.shape)
        centerLineNodeDisplacementsZ = np.zeros(centerLineNodeDisplacementsZUnordered.shape)
        for i in range(numFrames):
            centerLineNodeDisplacementsX[i,:] = centerLineNodeDisplacementsXUnordered[i,index_order]
            centerLineNodeDisplacementsY[i,:] = centerLineNodeDisplacementsYUnordered[i,index_order]
            centerLineNodeDisplacementsZ[i,:] = centerLineNodeDisplacementsZUnordered[i,index_order]
        
        # rearrange the node coordinats so they line up with the others. 
        centerLineNodeCoordinates = centerNodeCoordinates[index_order]
        
        #if the code has made it this far the FEA is complete
        break

    #Build the output dictionary
    data_output = {
        'odb_data':odb_data,
        'centerLineNodeCoordinates':centerLineNodeCoordinates,
        'centerLineNodeDisplacementsX':centerLineNodeDisplacementsX,
        'centerLineNodeDisplacementsY':centerLineNodeDisplacementsY,
        'centerLineNodeDisplacementsZ':centerLineNodeDisplacementsZ,
        'frequencies':freqData,
        'crack_length':crack_length,
        'mesh_deviationFactor':mesh_deviationFactor,
        'mesh_minSizeFactor':mesh_minSizeFactor,
        'mesh_size':mesh_size,
        'participationFactorX':np.asarray(odb_data['participationFactorX']),
        'participationFactorY':np.asarray(odb_data['participationFactorY']),
        'participationFactorZ':np.asarray(odb_data['participationFactorZ']),
        'effectiveMassX':np.asarray(odb_data['effectiveMassX']),
        'effectiveMassY':np.asarray(odb_data['effectiveMassY']),
        'effectiveMassZ':np.asarray(odb_data['effectiveMassZ']),
        'roller_location':roller_location,
        }
    
    # Go back up a level in the directy
    os.chdir('..')


    # delete all the files in the directy if asked for.
    if deleteFiles:
        files = glob.glob('abaqus_working_files/*')
        
        # if requested, skip the .simdir folder as can be written over.
        if ignoreSimdir:
            for i in range(len(files)):
                if files[i].endswith(".simdir"):
                    files.pop(i)
                    break
                   
        # delete the files           
        for f in files:
            try:
                removeDataFiles(f,sleepTime=1)
                # use "removeDataFiles" instead of "os.remove(f)" as
                # "removeDataFiles" will retry the delete.
            except:
                print('could not delete '+str(f))
                   
    return(data_output)
        

def scaleModeShapes(dataIn):  
    """ 
    Takes the mode coordinates, displacements, and participaton factor and 
    returns 
    
    """
    # this is a copy of the last function, but with a diminsion removed. I think they should be combined
    
    # extract the information of intrest, and orgnize such that the fist diminsion
    # of the array is the data in the test structure
    testModeCoordXIn = np.asarray(dataIn['centerLineNodeCoordinates'])[:,0]
    testModeDispZIn = np.copy(np.asarray(dataIn['centerLineNodeDisplacementsZ'])[:,:].T) # this index starts
                    # at mode 0 for the displacement shape
    testPartFactorZ = np.asarray(dataIn['participationFactorZ'])[:,1] # note that
                    # this index starts at 1, not 0 like the displacement
    testPartFactorZ = np.hstack((1,testPartFactorZ))# add a column of ones to the first 
                    # column so the partisipation factor for the displacment is 1.
    
    # The modes some times are positive, and sometimes negative, so to account for 
    # this the mode displacements are multiplied by the participation factors so 
    # they are always postive, than the participation factors are chnged to be postitive
    for ii in range(testPartFactorZ.shape[0]):
            # set all the mode shapes to act in the same direction
            testModeDispZIn[:,ii] = testModeDispZIn[:,ii] * np.sign(testPartFactorZ[ii])
            
            # scale all the mode shape to one. The -20 was added so mode shapes are not 
            # scaled off the tail for mode shape #7.
            testModeDispZIn[:,ii] = testModeDispZIn[:,ii]/np.max(np.abs(testModeDispZIn[:,ii]))
            # the abs call is for mode shape 0, as the sign on the partisipation factor may not be right. 
            # however, at this point all other mode shapes should be positive
            
            # make the participation factors all positive
            testPartFactorZ[ii] = testPartFactorZ[ii] * np.sign(testPartFactorZ[ii])
            
    # next the mode shapes need to be interpolated onto a common x space. 
    testModeCoordX = np.linspace(0, np.max(testModeCoordXIn), num=200, endpoint=True)
    
    testModeDispZ = np.zeros((testModeCoordX.shape[0],testModeDispZIn.shape[1]))
    
    for ii in range(testModeDispZIn.shape[1]):
        # y = trueModeDispZIn[i,:,ii]
        # x = trueModeCoordXIn[i,:]
        f = sp.interpolate.interp1d(testModeCoordXIn[:], testModeDispZIn[:,ii], kind='linear')
        testModeDispZ[:,ii] = f(testModeCoordX)
    
    
    dataOut = {
        'modeDispZ':testModeDispZ,
        'modeCoordX':testModeCoordX,
        'partFactorZ':testPartFactorZ,
        'freq':np.asarray(dataIn['frequencies']),
        }

    return(dataOut)

# define the direct mode based approach as given in "a probabilistic model  updating
# algorith for fatibue damage detection in aluminum hull structures"
def DMBA(trueOmega,testOmega,truePhi,testPhi,alpha):
    """ direct mode-based approach 
    
     define the direct mode based approach as given in "a probabilistic model  updating
     algorith for fatigue damage detection in aluminum hull structures
     """
        
    # MAC
    def MAC_i(truePhi,testPhi):
        x = np.abs(truePhi.T@testPhi)**2/((truePhi.T@truePhi)*(testPhi.T@testPhi))
        return(x)

    J_1=0
    for i in range(trueOmega.shape[0]):
        J_1 = J_1+((trueOmega[i]-testOmega[i])/testOmega[i])**2
    
    J_2=0
    for i in range(trueOmega.shape[0]):
        J_2 = J_2 + ((1-np.sqrt(MAC_i(truePhi[:,i],testPhi[:,i])))**2)/MAC_i(truePhi[:,i],testPhi[:,i])
    J = J_1 + alpha * J_2
    return J, J_1, J_2


def flexMatrix(ModeDispZ,Freq,mass_constants,Z_modes):
    """
    This is equation 11 in "A Probabilistic Model Updating Algorithm for Fatigue
    Damage Detection in Aluminum Hull Structures"
    """
    F = 0
    for i in Z_modes:
        FunPhi = np.expand_dims(ModeDispZ[:,i],axis=1)
        FunFreq = Freq[i-1]
        FunMass = mass_constants[i-1]
        F=F+(FunMass/FunFreq)**2*FunPhi@FunPhi.T  
    return(F)


def flexibilityBasedApproach (trueModeDispZ,trueFreq,trueMmassConstants,testModeDispZ,testFreq,testMmassConstants,Z_modes):
    """
    This is the Flexibility-Based Approach from "A Probabilistic Model Updating Algorithm for Fatigue
    Damage Detection in Aluminum Hull Structures"
    """
        
    # This is equation 11
    flexTrue = flexMatrix(trueModeDispZ,trueFreq,trueMmassConstants,Z_modes)
    flexTrial = flexMatrix(testModeDispZ,testFreq,testMmassConstants,Z_modes)
    
    flexDelta = flexTrue - flexTrial
    
    # caclulate the Frobenious norm of the delta flex matrix. equation 13
    flexNorm = np.linalg.norm(flexDelta,ord='fro')
    
    return(flexNorm)
