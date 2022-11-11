# Simulation
This page documents the simulation module. The simulator allows the control of a single agent while replaying the trajectories of all other vehicles from the dataset. To do so, the simulator requires the definition of a dataset, a dynamics model, a suite of sensors, an observer that map sensor measurements to agent observations, and a reward model. 

## Dataset
We convert the raw track data stored in pandas dataframe into a stack of frames, where each frame contains the state variables of all vehicles in the frame. The state variables contain: ``[x, y, vx, vy, ax, ay, psi_rad, length, width, track_id]``. We also collect the ego track id and the start and end frame of each episode. These information are stored in the ``VehicleTrajectories`` object. 

## Dynamics model
We use the constant acceleration model to step the simulator forward in time. Let $x$, $v$, and $a$ be the agent's position, velocity, and acceleration, the constant acceleration model updates position and velocity as:
* $x' = x + v * dt + a * dt^2$
* $v' = a * dt$

## Sensors
We have currently implemented three sensors. All sensors require the ``MapReader`` object as input. We define left of the ego vehicle and counter clockwise as positive. 
* ``EgoSensor``: This sensor tracks ego vehicle's lateral lane offset, ego longitudinal and lateral speed, ego heading error w.r.t. the lane centerline, lane centerline curvature, and lane id. 
* ``LeadVehicleSensor``: This sensor tracks the ego vehicle's relative distance, speed, and inverse tau w.r.t. the lead vehicle. These values are positive when the lead vehicle is in front of the ego vehicle. It also tracks lead vehicle's lateral lane offset, lane speed, and lead vehicle track id. 
* ``LidarSensor``: This sensor simulates a lidar scanner. It divides the ego vehicle's surrounding area into ``num_beams`` number of equally sized fan-shaped bin. For each bin, it finds the closest other vehicle and computes and range and range rate (speed of change of range) to that vehicle. If there is no other vehicle within a maximum distance of ``max_range``, it returns ``max_range`` as the range and ``0`` as the range rate. 

All sensors have the ``get_obs`` method which takes in the state of ego and all other vehicles and returns the measurements and positions of the vehicle hit by the sensor beam. Hit positions are used to create animations. 

## Observer
The observer object interfaces the simulator with the agent. It takes all sensor measurements stored in a dictionary and flatten it into a $1 \times d$ vector. This vector becomes the agent's observation. Once the agent emits a control action, the observer converts it from the frenet coordinate into the cartesian coordinate before passing it to the simulator. 

We have implemented a special ``CarfollowObserver``. In this observer, we expect the agent to only emit the longitudinal control. Lateral control is handeled by the observer itself using a simple feedback control rule. These two control signals are concatenated before passing to the simulator. 

The observer also computes the ``info`` output field in the simulator. Currently, the ``CarfollowObserver`` will set ``info["terminate"] = True`` if the ego vehicle is ahead of the lead vehicle for more than a certain distance and the ego vehicle's lane offset is higher than a certain value.

## Reward
The reward module monitors the quality of a simulated trajectory. Currently the reward module calculate the absolute distance between the simulated ego vehicle position and the position of the ego vehicle in the actual track. 

## Simulator
The ``InteractionSimulator`` follows the OpenAI Gym framework with the following methods:
* ``reset``: This method requires an argument of the episode id to initialize the state of the simulator to the first frame of the entered episode. It then computes and returns sensor measurements. 
* ``step``: This method takes in the agent's action, converts it to the global cartesian coordinate, steps the ego state forward using the dynamics model, and then computes sensor measurements. This method outputs ``obs, rwd, done, info``. ``done = True`` if the simulator has reached the last time step of the episode. 

The simulator also stores the simulated states, actions, observations, and rewards internally under the ``self._state`` dictionary. For specific fields in the dictionary please see code. 