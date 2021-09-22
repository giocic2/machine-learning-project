import gym
import numpy as np
import matplotlib.pyplot as plt

# Define environment.
env = gym.make("MountainCar-v0")

# Q-learning settings.
DISCOUNT = 0.95
LEARNING_RATE = 0.1
EPISODES = 10_000 + 1
SHOW_EVERY = 1000 # We render only a few episodes, not every single one.

# Statistics settings.
STATS_EVERY = 100
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Q-table saving setting
SAVE_EVERY = 10

# Exploration settings.
epsilon = 1 # not a constant, going to be decayed. The higher epsilon, the more random action we will try.
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Some parameters of the environment.
print("Upper limit to observation state: ", env.observation_space.high)
print("Lower limit to observation state: ",env.observation_space.low)
print("Number of possible actions: ", env.action_space.n) # 0:"go left", 1:"stay", 2:"go right".

# Q table, let's make the observation space discrete.
DISCRETE_OS_SIZE = [20,20] # Discrete observation space size.
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print("Discrete O.S. window size: ", discrete_os_win_size)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))

# Q table initialization with random values.
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # 20x20x3 size (every possible action for every possible state (in the buckets).

successes = 0
last_1000_successes = 0

for episode in range(EPISODES): # We need a lot of episodes to train.
    print("Episode: ", episode)
    state = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0 # For each episode, is the sum of all rewards obtained.

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:
        
        # Pick randomly if we do exploration or not.
        if np.random.random() > epsilon:
            # Get action from Q table.
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action) # Underscore it's convention for throwaway variables.
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render == True:
            env.render()   

        # Update Q-value, if simulation not ended.
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position: # If goal position achieved, end the simulation.
            q_table[discrete_state + (action,)] = 0 # Reward for completing task.
            print("GOAL ACHIEVED!")
            successes += 1
            if episode > (EPISODES - 1000):
                last_1000_successes += 1

        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards["max"].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards["min"].append(min(ep_rewards[-STATS_EVERY:]))
        print(f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")
    # Saving q-table each episode, or form time to time
    if episode % SAVE_EVERY == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
env.close()
print("Total number of successes: ", successes)
print("Number of successes on last 1000 episodes: ", last_1000_successes)

# Plot statistics.
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=1)
plt.show()