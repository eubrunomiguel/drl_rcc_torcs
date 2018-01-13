import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self):
        pass

    def act(self, ob, reward, done):

        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values
        #
        #
        #
        #  and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]

        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision, trackPos = ob

        """ The code below is for checking the vision input. This is very heavy for real-time Control
            So you may need to remove.
        """

        img = np.ndarray((64,64,3))
        for i in range(3):
            img[:, :, i] = 255 - vision[:, i].reshape((64, 64))

        plt.imshow(img, origin='lower')
        plt.draw()
        plt.pause(0.001)

        # random action
        return np.tanh(np.random.randn(1))
