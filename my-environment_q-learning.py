import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt

style.use("ggplot")

# Q-table variables.
TIME_BUCKETS = 10 # Time index.
FREQ_BUCKETS = 20 # VCO frequency.
TEMP_BUCKETS = 2 # VCO temperature.
ACTIONS_NUMBER = 3

# Environment variables.
START_TIME = 0 # seconds
END_TIME = 10e-3 # seconds
time_step = (END_TIME - START_TIME) / TIME_BUCKETS # sec/bucket
MIN_FREQ = 22e9 # Hz
MAX_FREQ = 26e9 # Hz
freq_step = (MAX_FREQ - MIN_FREQ) / FREQ_BUCKETS # Hz/bucket

# Q-learning settings.
EPISODES_LIMIT = 10_000 + 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = np.full(TEMP_BUCKETS, 0.5)
EPS_DECAY = np.full(TEMP_BUCKETS, 0.9998) # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 500
Q_VALUE_MIN = -5
Q_VALUE_MAX = 0

# The state of a modulation is defined as a pair of coordinates in the time-frequency 2D space.
class modulation_state:
    def __init__(self):
        self.freq = 0
    def __str__(self):
        return f"{self.time}, {self.freq}"
    def action(self, choice):
        if choice == 0: 
            self.freq = self.freq # keep same frequency
        if choice == 1: 
            self.freq += 1 # increase frequency
        if choice == 2:
            self.freq -= 1 # decrease frequency

        # If we are out of bounds, fix!
        if self.freq > (FREQ_BUCKETS-1):
            self.freq = FREQ_BUCKETS - 1
        elif self.freq < 0:
            self.freq = 0

def getTemperature():
    if np.random.random() > 0:
        temperature_bucket= 0
    else: 
        temperature_bucket = 1
    return temperature_bucket

# Let's define goal modulation, the one that must be learned from the learning agent.
goal_modulation = np.ndarray((TEMP_BUCKETS, TIME_BUCKETS))
goal_modulation[0,:] = [0,1,2,3,3,4,4,5,5,5] # Modulation @ 1st temperature value.
goal_modulation[1,:] = [1,3,3,4,4,5,5,6,6,6] # Modulation @ 2nd temperature value.
# Plot goal modulation
for tempIndex in range(TEMP_BUCKETS):
    plt.plot([i for i in range(TIME_BUCKETS)], goal_modulation[tempIndex,:])
plt.ylabel("Frequency bucket")
plt.xlabel("Time bucket")
plt.show()

# Q-table: random initialization
q_table = np.random.uniform(low=Q_VALUE_MIN, high=Q_VALUE_MAX, size=(ACTIONS_NUMBER,TEMP_BUCKETS,TIME_BUCKETS,FREQ_BUCKETS))

rewards_history = []

for episode in range(EPISODES_LIMIT):
    time_bucket = 0
    current_temperature = getTemperature()
    learning_agent = modulation_state()
    episode_reward = 0 # Reset this temporary value at every episode.
    if episode % SHOW_EVERY == 0:
        print(f"On episode #{episode}, epsilon is {epsilon}")
        # Show how the training is going on... te be implemented
        show = True
    else:
        show = False
    while time_bucket < (TIME_BUCKETS-1):
        if np.random.random() > epsilon[current_temperature]: # exploiting action
            action = np.argmax(q_table[:, current_temperature, time_bucket, learning_agent.freq])
        else: # explorative action
            action = np.random.randint(0, ACTIONS_NUMBER)
        
        # Current state
        current_time = time_bucket
        current_freq = learning_agent.freq
        
        # Take the action!
        learning_agent.action(action)
        # Evaluate reward for this time step.
        reward = - abs(learning_agent.freq - goal_modulation[current_temperature,time_bucket])
        # Time has passed
        time_bucket += 1

        # New state
        new_time = time_bucket
        new_freq = learning_agent.freq

        # Evaluate new Q-value
        max_future_q = np.max(q_table[:, current_temperature, new_time, new_freq])
        current_q = q_table[action, current_temperature, current_time, current_freq]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[action, current_temperature, current_time, current_freq] = new_q
        episode_reward += reward
    rewards_history.append(episode_reward)
    epsilon[current_temperature] *= EPS_DECAY[current_temperature]
moving_avg = np.convolve(rewards_history, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()