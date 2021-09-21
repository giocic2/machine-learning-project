import gym
import numpy as np

# Define environment
env = gym.make("MountainCar-v0")

# Q-learning settings
DISCOUNT = 0.95
LEARNING_RATE = 0.1
EPISODES = 10_000
SHOW_EVERY = 1_000 # We render only a few episodes, not every single one

# Exploration settings
epsilon = 1 # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Some parameters of the environment
print("Upper limit to observation state: ", env.observation_space.high)
print("Lower limit to observation state: ",env.observation_space.low)
print("Number of possible actions: ", env.action_space.n) # 0:"go left", 1:"stay", 2:"go right"

# Q table, let's make the observation space discrete
DISCRETE_OS_SIZE = [20,20] # observation
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print("Discrete O.S. window size: ", discrete_os_win_size)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))

# Q table initialization with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # 20x20x3 size (every possible action for every possible state (in the buckets).


for episode in range(EPISODES): # We need a lot of episodes to train
    state = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    print("Episode: ", episode)

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:
        
        # Pick randomly if we do exploration or not
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action) # underscore it's convention for throwaway variables.
        # print("Reward: ", reward, "; New State: ", new_state)
        new_discrete_state = get_discrete_state(new_state)
        if render == True:
            env.render()   

        # Update Q-value, if simulation not ended
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position: # If goal position achieved, end the simulation
            q_table[discrete_state + (action,)] = 0 # Reward for completing task
            print("GOAL ACHIEVED!")
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        
env.close()
