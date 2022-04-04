# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__


import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np
import pickle as pickle
import json as json
import os


#%% Unpack the JSON

with open('data_in.json') as f:
    D = json.load(f)
f.close()
    
crack_length = D['crack_length']
roller_location =D['roller_location']
mesh_deviationFactor = D['mesh_deviationFactor']
mesh_minSizeFactor = D['mesh_minSizeFactor']
mesh_size = D['mesh_size']
crack_width = D['crack_width']


#%% build the model


s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.rectangle(point1=(0.0, 0.0), point2=(0.75, 0.0254))
p = mdb.models['Model-1'].Part(name='beam', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['beam']
p.BaseShell(sketch=s1)
s1.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['beam']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']

p = mdb.models['Model-1'].parts['beam']
f, e = p.faces, p.edges
t = p.MakeSketchTransform(sketchPlane=f[0], sketchUpEdge=e[1], 
    sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0))
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=1.52, 
    gridSpacing=0.03, transform=t)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Model-1'].parts['beam']
p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
s.rectangle(point1=(0.005, 0.0127-(crack_length/2)), point2=(0.005+crack_width, 0.0127+
                                                            (crack_length/2)))

p = mdb.models['Model-1'].parts['beam']
f1, e1 = p.faces, p.edges
p.CutExtrude(sketchPlane=f1[0], sketchUpEdge=e1[1], sketchPlaneSide=SIDE1, 
    sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']

session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)

#%% define the material 

mdb.models['Model-1'].Material(name='Aluminum')
mdb.models['Model-1'].materials['Aluminum'].Density(table=((2700.0, ), ))
mdb.models['Model-1'].materials['Aluminum'].Elastic(table=((70000000000.0, 
    0.32), ))

#%% create the sections 
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['beam']
a.Instance(name='beam-1', part=p, dependent=ON)
p1 = mdb.models['Model-1'].parts['beam']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['Model-1'].HomogeneousShellSection(name='Section-1', 
    preIntegrate=OFF, material='Aluminum', thicknessType=UNIFORM, 
    thickness=0.01, thicknessField='', nodalThicknessField='', 
    idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
    thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
    integrationRule=SIMPSON, numIntPts=5)
p = mdb.models['Model-1'].parts['beam']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
region = p.Set(faces=faces, name='Set-1')
p = mdb.models['Model-1'].parts['beam']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)






#%% partition the face

mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.03, name='__profile__', 
    sheetSize=1.52, transform=
    mdb.models['Model-1'].parts['beam'].MakeSketchTransform(
    sketchPlane=mdb.models['Model-1'].parts['beam'].faces[0], 
    sketchPlaneSide=SIDE1, 
    sketchUpEdge=mdb.models['Model-1'].parts['beam'].edges[5], 
    sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0)))
mdb.models['Model-1'].parts['beam'].projectReferencesOntoSketch(filter=
    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-1.0, 0.0127), 
    point2=(1.0, 0.0127))
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[10])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(roller_location, -1.0), point2=(
    roller_location, 1.0))
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
    False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].parts['beam'].PartitionFaceBySketch(faces=
    mdb.models['Model-1'].parts['beam'].faces.getSequenceFromMask(('[#1 ]', ), 
    ), sketch=mdb.models['Model-1'].sketches['__profile__'], sketchUpEdge=
    mdb.models['Model-1'].parts['beam'].edges[5])
del mdb.models['Model-1'].sketches['__profile__']


#%% mesh the parts

mdb.models['Model-1'].parts['beam'].seedPart(deviationFactor=mesh_deviationFactor, 
    minSizeFactor=mesh_minSizeFactor, size=mesh_size)
mdb.models['Model-1'].parts['beam'].generateMesh()


#%% create steps

mdb.models['Model-1'].StaticStep(name='Static', previous='Initial')

# create the mass normilization factor, this create the parameter
# normalization=mass in the input file
mdb.models['Model-1'].FrequencyStep(limitSavedEigenvectorRegion=None, name=
    'Frequency', numEigen=20, previous='Static')

#%% create BC
mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['beam-1'].edges.getSequenceFromMask(
    ('[#420 ]', ), ), name='Set-1')
mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial', 
    distributionType=UNIFORM, fieldName='', localCsys=None, name='Fixed', 
    region=mdb.models['Model-1'].rootAssembly.sets['Set-1'], u1=SET, u2=SET
    , u3=SET, ur1=SET, ur2=SET, ur3=SET)


mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['beam-1'].edges.getSequenceFromMask(
    ('[#180 ]', ), ), name='Set-2')
mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Static', 
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'pinned', region=mdb.models['Model-1'].rootAssembly.sets['Set-2'], u1=UNSET, 
    u2=UNSET, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)


#%% create the Set for the line down the middle

mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['beam-1'].edges.getSequenceFromMask(
    ('[#4011 ]', ), ), name='Set-3')


#%% Apply the gravity load

mdb.models['Model-1'].Gravity(comp3=-9.81, createStepName='Static', 
    distributionType=UNIFORM, field='', name='Gravity')


#%% create a element set for the entire model

# I could not figure out how to make a set of all of the elements, so I used
# the mask command, however, the mask may change and mess this up.
mdb.models['Model-1'].rootAssembly.regenerate()
mdb.models['Model-1'].rootAssembly.Set(elements=
    mdb.models['Model-1'].rootAssembly.instances['beam-1'].elements.getSequenceFromMask(
    mask=('[#ffffffff:14 #1fffffff ]', ), ), name='Set-4')


#%% Run the job

mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
 	atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
 	memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
 	explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
 	modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
 	scratch='', resultsFormat=ODB)
mdb.jobs['Job-1'].submit(consistencyChecking=OFF)


# Wait for completion, as the result file will generated only after running,
# so code should stop here
mdb.jobs['Job-1'].waitForCompletion()

#%% Extract the data

from odbAccess import *



# Read the odb file, change name and path as per your requirement
odb = openOdb(path='Job-1.odb', readOnly=True) 
    

# extract basic info about the model
numNodes = len(odb.rootAssembly.instances['BEAM-1'].nodes)
numElements = len(odb.rootAssembly.instances['BEAM-1'].elements)

# Create a step 
step1 = odb.steps['Frequency']

# Read the history region, the key for this call 
# determines what you are picking as your region, 
# it can be a node, an element or as in this case, 
# the whole model. 
region = step1.historyRegions['Assembly ASSEMBLY']
    
# Once the history region has been set, all the history outputs can 
# be referenced using their names. EIGFREQ for frequency etc. 
freqData = region.historyOutputs['EIGFREQ'].data

participationFactorX = region.historyOutputs['PF1'].data
participationFactorY = region.historyOutputs['PF2'].data
participationFactorZ = region.historyOutputs['PF3'].data

# extract the displacement components of the effective mass for the entire model
effectiveMassX = region.historyOutputs['EM1'].data
effectiveMassY = region.historyOutputs['EM2'].data
effectiveMassZ = region.historyOutputs['EM3'].data

# extract the labels and coordinates for the nodes on the centerline 
nodeSet4 = odb.rootAssembly.nodeSets['SET-3']
centerNodeLabels = []
centerNodeCoordinates = []
for q in range(len(nodeSet4.nodes[0])):
 	centerNodeLabels.append(nodeSet4.nodes[0][q].label)

for q in range(len(nodeSet4.nodes[0])):
    centerNodeCoordinates.append([float(nodeSet4.nodes[0][q].coordinates[0]),
                                  float(nodeSet4.nodes[0][q].coordinates[1]),
                                  float(nodeSet4.nodes[0][q].coordinates[2])])
    
# extract the displacement data for each node.    
frame = []
for i_frame in odb.steps['Frequency'].frames:
    lastFrame = i_frame#odb.steps['Frequency'].frames[0]
    displacement = lastFrame.fieldOutputs['U']
    center = odb.rootAssembly.instances['BEAM-1'].nodeSets['SET-1']
    centerDisplacement = displacement.getSubset(region=center)
    modeNodes = []
    modeDisplacement = []
    for v in centerDisplacement.values:
        modeNodes.append(v.nodeLabel)
        # I has an issue with the np.float 32 format crashing the json. so this code
        # was updated with the float commands.
        modeDisplacement.append([float(v.data[0]),float(v.data[1]),float(v.data[2])])     
   
    frame_data = {'modeNodes':modeNodes,'modeDisplacement':modeDisplacement}
    
    frame.append(frame_data)


#%% Save the data to a JSON


data = {'freqData':freqData,
        'centerNodeLabels':centerNodeLabels,
        'frame':frame,
        'centerNodeCoordinates':centerNodeCoordinates,
        'participationFactorX':participationFactorX,
        'participationFactorY':participationFactorY,
        'participationFactorZ':participationFactorZ,
        'effectiveMassX':effectiveMassX,
        'effectiveMassY':effectiveMassY,
        'effectiveMassZ':effectiveMassZ,
        'numNodes':numNodes,
        'numElements':numElements,}
   
with open('data_out.json', 'w') as f:
    json.dump(data, f)
f.close()    

#%% Notes


"""

To export the K and M matrix, you can add the following code to the input file.
More can be found at the following links, there is example input file code
at the bottom of the first link

https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-mtxgenerationperturbation.htm
https://abaqus-docs.mit.edu/2017/English/SIMACAEOUTRefMap/simaout-c-format.htm


** ----------------------------------------------------------------
** 
** STEP: matrix
** 
*STEP
*MATRIX GENERATE, STIFFNESS, MASS, VISCOUS DAMPING,
STRUCTURAL DAMPING, LOAD
*MATRIX OUTPUT, STIFFNESS, MASS, VISCOUS DAMPING,
STRUCTURAL DAMPING, LOAD, FORMAT=COORDINATE
**
*End Step

This code will parse the data exported to a .mtx file into a numpy array. 

# packages needed
# import numpy as np
# import scipy as sp

# load the data
A_sparse = np.loadtxt('Job-1_MASS2.mtx')
i = A_sparse[:, 0].astype(np.int)
j = A_sparse[:, 1].astype(np.int)
M = i.max()
N = j.max()
# Python uses 0 based index! so we'll subtract 1 from our row and column index
A = sp.sparse.coo_matrix((A_sparse[:, 2], (i-1, j-1)), shape=(M, N))
# to use a numpy array instead of the sparse matrix format
M = A.toarray()


pickle.dump(M,open('M.pkl','wb'))


"""



