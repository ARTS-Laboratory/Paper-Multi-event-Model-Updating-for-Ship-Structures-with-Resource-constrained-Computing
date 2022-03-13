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

## Code FEA/Python:
* Main goal: Construct the FEA model while using the least amount of computational resources
* Transform completed model into code. 
* Change model code. 
* Submit changed model to Abaqus solver.
* Create surface plot of altered models.
* Run particle swarm on surface plot to determine global min.
* Update FEA model
* Automate the above steps

To construct the FEA model and easily change any aspect of the model while using the least amount of computational resources, the model construction was simplified to a 2-D model and was recorded with the Macros option in Abaqus. This converts the user interface commands into code that the Abaqus solver uses to compute the chosen results. Once this code was saved to .txt file python was used to submit the model to the Abaqus solver with altered roller locations and fatigue crack length. Solving the altered models this way save computational resources since the model is solved without the user interface ever loading.  

For the particle swarm iteration # and particle #, an optimal combination was found to be 25 and 10 respectively. This was found by testing all combinations until a 40 iteration and 20 particle combination was achieved. The error between the returned global min and true gloabl min is computed and plotted with iteration # and particle #. This is shown in Figure 3. The chosen optimal combination is at the beginning of the plateau since any combination after will return the same global min while taking more time and computational resources. 

<center>

![image](https://user-images.githubusercontent.com/69403619/158040465-5c020404-822a-47da-9fda-625bfa3443fa.png)
      Figure 3. Particle Swarm optimal iteration and particle combination

</center>



## Results:
