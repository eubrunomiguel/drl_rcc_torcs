
import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class Environment:
    default_speed = 10
    initial_reset = True

    def __init__(self, throttle=False, gear_change=False):
        self.throttle = throttle

        self.gear_change = gear_change

        self.initial_run = True

        self.reset_torcs()

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        # action_torcs is a reference object to the mutators
        action_torcs = self.client.R.d

        # Steering
        action_torcs['steer'] = u[0]  # steering in [-1, 1]

        # Adjust speed
        self.adjust_speed()

        # Adjust gear
        self.adjust_gear()

        # Apply the Agent's action into torcs
        self.client.respond_to_server()

        # Get the response of TORCS
        self.client.get_servers_input()

        # Get the current full-observation from torcs
        obs = self.client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Calculate Reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # Termination judgement #########################
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            self.client.R.d['meta'] = True

        # Episode is terminated if the agent runs backward
        if np.cos(obs['angle']) < 0:
            self.client.R.d['meta'] = True

        # Send a reset signal
        if self.client.R.d['meta'] is True:
            self.initial_run = False
            self.client.respond_to_server()

        self.time_step += 1

        return self.observation, reward, self.client.R.d['meta']

    def adjust_gear(self):
        self.client.R.d['gear'] = 1

    def adjust_speed(self):
        # if self.client.S.d['speedX'] < self.default_speed - (self.client.R.d['steer'] * 50):
        #     self.client.R.d['accel'] += .01
        # else:
        #     self.client.R.d['accel'] -= .01
        #
        # if self.client.R.d['accel'] > 0.2:
        #     self.client.R.d['accel'] = 0.2
        #
        # if self.client.S.d['speedX'] < 10:
        #     self.client.R.d['accel'] += 1 / (self.client.S.d['speedX'] + .1)
        #
        # # Traction Control System
        # if ((self.client.S.d['wheelSpinVel'][2] + self.client.S.d['wheelSpinVel'][3]) -
        #         (self.client.S.d['wheelSpinVel'][0] + self.client.S.d['wheelSpinVel'][1]) > 5):
        #     action_torcs['accel'] -= .2

        # be constant here
        if self.client.S.d['speedX'] < self.default_speed:
            self.client.R.d['accel'] = 1
        else:
            self.client.R.d['accel'] = 0

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=True)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()  # Get the initial input from torcs

        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.observation

    def close_torcs(self):
        os.system('pkill torcs')

    def reset_torcs(self):
        self.close_torcs()
        time.sleep(0.5)
        os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        time.sleep(1)
        os.system('sh autostart.sh')
        time.sleep(1)

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'opponents',
                 'rpm',
                 'track',
                 'wheelSpinVel',
                 'img', 'trackPos']
        Observation = col.namedtuple('Observaion', names)

        # Get RGB from observation
        image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

        return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                           speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                           opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                           track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                           wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                           img=image_rgb, trackPos=np.array(raw_obs['trackPos'], dtype=np.float32))
