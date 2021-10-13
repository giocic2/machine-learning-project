import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt

# Resolutions
TIME_RESOL = 1e-3 # s
FREQ_RESOL = 10 # Hz
VOLTAGE_RESOL = 100e-3 # V
TEMP_RESOL = 10 # K

# Q-table and environment variables.
# Number of possible actions for the learning agent
ACTIONS_NUMBER = 3
# Time
START_TIME = 0 # seconds
END_TIME = 10e-3 # seconds
time_buckets = round((END_TIME - START_TIME) / TIME_RESOL) + 1
time_axis = np.linspace(start=START_TIME, stop=END_TIME, num=time_buckets, endpoint=True)
# Frequency
MIN_FREQ = 0 # Hz
MAX_FREQ = 100 # Hz
freq_buckets = round((MAX_FREQ - MIN_FREQ) / FREQ_RESOL) + 1
freq_axis = np.linspace(start=MIN_FREQ, stop=MAX_FREQ, num=freq_buckets, endpoint=True)
# Temperature
MIN_TEMP_C = 0
MAX_TEMP_C = 50
min_temp = MIN_TEMP_C + 273.15 # K
max_temp = MAX_TEMP_C + 273.15 # K
temp_buckets = round((max_temp - min_temp) / TEMP_RESOL) + 1
temp_axis = np.linspace(start=min_temp, stop=max_temp, num=temp_buckets, endpoint=True)
# Voltage
MIN_VOLTAGE = 0.5 # Volt
MAX_VOLTAGE = 5 # Volt
voltage_buckets = round((MAX_VOLTAGE - MIN_VOLTAGE) / VOLTAGE_RESOL) + 1
voltage_axis = np.linspace(start=MIN_VOLTAGE, stop=MAX_VOLTAGE, num=voltage_buckets, endpoint=True)


# Q-learning settings.
EPISODES_LIMIT = 10_000 + 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = np.full(temp_buckets, 0.5)
EPS_DECAY = np.full(temp_buckets, 0.9998) # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1_000
Q_VALUE_MIN = -5
Q_VALUE_MAX = 0

class instant_frequency:
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
        if self.freq > (freq_buckets-1):
            self.freq = freq_buckets - 1
        elif self.freq < 0:
            self.freq = 0

def getTemperature():
    if np.random.random() > 0:
        temperature_bucket= 0
    else: 
        temperature_bucket = 1
    return temperature_bucket

# Model of the VCO tuning law
tuning_law = np.ndarray((temp_buckets, voltage_buckets))
# VCO tuning law: coefficients
a1 = 1.969e9 # Hz, dynamic of the log function
a2 = 23.947e9 # Hz, offset for the log function
a3 = 10e6 # Hz/K, temperature dependence
a4 = 233 # K, offset for the temperature drift
temp_index = 0
voltage_index = 0
for temp in temp_axis:
    for voltage in voltage_axis:
        tuning_law[temp_index,voltage_index] = a1 * np.log(voltage) + a2 - (a3 * (temp - a4))
        voltage_index += 1
    temp_index += 1
    voltage_index = 0
# Plot VCO tuning law
for temp_index in range(temp_buckets):
    plt.plot(voltage_axis, tuning_law[temp_index,:])
plt.ylabel("Frequency [GHz]")
plt.xlabel("Voltage [V]")
plt.show()

# # Let's define goal modulation, the one that must be learned from the learning agent.
# goal_modulation = np.ndarray((temp_buckets, time_buckets))
# # Example goal modulation
# for temp in range(temp_buckets):
#     for time in range(time_buckets):
#         goal_modulation[temp,time] = (time+3)**(1/(3+temp)) + 1/(temp+1)
#     goal_modulation[temp,:] = goal_modulation[temp,:] * freq_buckets / 3 # scale modulation on all axis
# # Plot goal modulation
# for tempIndex in range(temp_buckets):
#     plt.plot([i for i in range(time_buckets)], goal_modulation[tempIndex,:])
# plt.ylabel("Frequency bucket")
# plt.xlabel("Time bucket")
# plt.show()

# # Q-table: random initialization
# q_table = np.random.uniform(low=Q_VALUE_MIN, high=Q_VALUE_MAX, size=(ACTIONS_NUMBER,temp_buckets,time_buckets,freq_buckets))

# # Q-learning process
# rewards_history = []
# VCO_history = np.zeros(time_buckets)

# for episode in range(EPISODES_LIMIT):
#     time_bucket = 0
#     current_temperature = getTemperature()
#     VCO = instant_frequency()
#     VCO_history[time_bucket] = VCO.freq
#     episode_reward = 0 # Reset this temporary value at every episode.

#     while time_bucket < (time_buckets-1):
#         if np.random.random() > epsilon[current_temperature]: # exploiting action
#             action = np.argmax(q_table[:, current_temperature, time_bucket, VCO.freq])
#         else: # explorative action
#             action = np.random.randint(0, ACTIONS_NUMBER)
        
#         # Current state
#         current_time = time_bucket
#         current_freq = VCO.freq
        
#         # Take the action!
#         VCO.action(action)
#         # Evaluate reward for this time step.
#         reward = - abs(VCO.freq - goal_modulation[current_temperature,time_bucket])
#         # Time has passed
#         time_bucket += 1

#         # New state
#         new_time = time_bucket
#         new_freq = VCO.freq
#         VCO_history[time_bucket] = VCO.freq

#         # Evaluate new Q-value
#         max_future_q = np.max(q_table[:, current_temperature, new_time, new_freq])
#         current_q = q_table[action, current_temperature, current_time, current_freq]
#         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#         q_table[action, current_temperature, current_time, current_freq] = new_q
#         episode_reward += reward
#     if episode % SHOW_EVERY == 0:
#             print(f"On episode #{episode}, epsilon is {epsilon}")
#             # Show how the training is going on
#             plt.plot([i for i in range(time_buckets)], goal_modulation[current_temperature,:])
#             plt.plot([i for i in range(time_buckets)], VCO_history[:])
#             plt.ylabel("Frequency bucket")
#             plt.xlabel("Time bucket")
#             plt.show()
#     rewards_history.append(episode_reward)
#     epsilon[current_temperature] *= EPS_DECAY[current_temperature]
# moving_avg = np.convolve(rewards_history, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"Reward {SHOW_EVERY}ma")
# plt.xlabel("episode #")
# plt.show()