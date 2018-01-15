# drl_rcc_torcs

### Options

* End-to-end neural network
  * Run the simulator with the "perfect" driver. Collect the images and the steering angle.
  * Create a network based on VGG
  * Error function.
  * Training
* Actor/Critic
* Segmentation

### Links

* Simulated Car Racing Championship Competition Software Manual
  * https://arxiv.org/abs/1304.1672
  * https://arxiv.org/pdf/1304.1672.pdf
* Gym-TORCS
  * https://github.com/ugo-nama-kun/gym_torcs
* Deep Reinforcement Learning approach to Autonomous Navigation
  * https://github.com/bhanuvikasr/Deep-RL-TORCS
* End-to-End Deep Learning for Self-Driving Cars
  * https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
* Teaching a Machine to Steer a Car
  * https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c
* Community Steering Models
  * https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models
* Reinforcement learning with docker and torcs
  * https://github.com/bn2302/rl_torcs

### Notes

* If you want to compile for yourself, use /home/gym_torcs/vtorcs-RL-color/src/libs/raceengineclient/raceengine.cpp
* to get from little screen to big screen: down, down, enter; down, enter; right, right, enter; enter
* run with command line --random option to enable random track selector

### todo
* preprocess image, image segmentation
* test network and solver
* adjust hyperparameters
