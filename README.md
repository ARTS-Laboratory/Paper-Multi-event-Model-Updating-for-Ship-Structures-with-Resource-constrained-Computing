# Code repository for the paper "Multi-event-Model-Updating-for-Ship-Structures-with-Resource-constrained-Computing"
General content for the SPIE paper titled Multi-event Model Updating for Ship Structures with Resource-constrained Computing
> **_REFERENCE:_** Jason Smith, Hung-Tien Huang, Austin Downey, Alysson Mondoro, Benjamin Grisso and Sourav Banerjee. Multi-event Model Updating for Ship Structures with
Resource-constrained Computing. In Proceedings of 2022 SPIE Smart Structures + Nondestructive Evaluation, 2022

## Introduction:
This project is under the Digital Twin umbrealla that specifically focuses on the “Automatic fusing of data with models", which simply put, the FEA model is automatically updated based on the output data of n constructed models. The importance and implementation is focused on reducing impacts and fatigue damage to ship components as its occurs while optimally utlizing and reconﬁguring the limited computational resources the ship has to offer. A simple example of optimally using the limited computational resources is when a ship experiences an impact (explosion, large wave load, collision etc.) and the computational reources are reconfigured to only calculate conditions of high importance such as remaining ship life and impact detection. Based on these calculated conditions actionable decisions are automatically made that include the closing of hatches and doors to reduce ship flooding and the relocation of machinary that is near the impact location. 

## FEA:
The FEA model is a “Smart Beam”, which is just a connection inside a ship. Some examples this can represent are a truss support connection, a section of the ship hull or a support beam. This beam tracks foreground and background changes such as impacts and cracks caused by fatigue. To implement the condition tracking of each condition the FEA model is subjected to a roller location change (reprensentative impact, since both are a sudden boundary condition change) and a growing linear fatigue crack near the beams left fixity as shown in Figure 1.  

<center>
  
![image](https://user-images.githubusercontent.com/69403619/158039826-71f1a82b-4392-4bee-983c-645953be15af.png)
      Figure 1. FEA model of smart beam

</center>

The fatigue crack start in the center of the beam and grows toward each edge. After the model is created, the difference in the true and trial Flexibility matricies is taken and plotted with its crack length and roller location to create a 3-D surface plot for the particle swarm to run on (Figure 2).

<center>

![Search_Space](https://user-images.githubusercontent.com/69403619/158040135-99144c21-2f0f-4cb2-803a-42448c6bee4e.png)
      Figure 2. Flexibility error surface plot for the Particle Swarm

</center>

## Code:
For the particle swarm

## Results:
