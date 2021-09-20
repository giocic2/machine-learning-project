import gym
import numpy as np

env = gym.make("MountainCar-v0")
print("Starting observation state (velocity, position): ", env.reset())

print("Upper limit to observation state: ", env.observation_space.high)
print("Lower limit to observation state: ",env.observation_space.low)
print(env.action_space.n) # 0:"go left", 1:"stay", 2:"go right"

# Q table, let's make the observation space discrete
DISCRETE_OS_SIZE = [20,20] # observation
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)
# Q table initialization with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # 20x20x3 size (every possible action for every possible state (in the buckets).

done = False

while not done:
    action = 2
    newState, reward, done, _ = env.step(action) # underscore it's convention for throwaway variables.
    print("Reward: ", reward, "; New State: ", newState)
    env.render()
    
env.close()
