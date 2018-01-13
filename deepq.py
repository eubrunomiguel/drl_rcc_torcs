from environment import Environment
from agent import Agent
import numpy as np

if __name__ == "__main__":
    episode_count = 10
    max_steps = 50

    env = Environment()
    agent = Agent()

    for i in range(episode_count):
        reward = 0
        done = False
        step = 0
        total_reward = 0.

        print("Episode : " + str(i))

        # Sometimes you need to relaunch TORCS because of the memory leak error
        if np.mod(i, 3) == 0:
            observation = env.reset(relaunch=True)
        else:
            observation = env.reset()

        for j in range(max_steps):
            action = agent.act(observation, reward, done)
            observation, reward, done = env.step(action)
            total_reward += reward
            step += 1

            if done:
                break

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close_torcs()  # This is for shutting down TORCS
    print("Finish.")
