import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()

# Resolutions
TIME_RESOL = 0.1e-3 # s
FREQ_RESOL = 10e6 # Hz
VOLTAGE_RESOL = 100e-3 # V
TEMP_RESOL = 10 # K

# Q-table and environment variables.
# Number of possible actions for the learning agent
ACTIONS_NUMBER = 3
# Time
START_TIME = 0 # seconds
END_TIME = 5e-3 # seconds
time_buckets = round((END_TIME - START_TIME) / TIME_RESOL) + 1
time_axis = np.linspace(start=START_TIME, stop=END_TIME, num=time_buckets, endpoint=True)
# Frequency
MIN_FREQ = 0 # Hz
MAX_FREQ = 120e6 # Hz
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
# FMCW linear modulation
BANDWIDTH = 3e9 # Hz
CHIRP_PERIOD = END_TIME # s
START_FREQUENCY = 23e9 # Hz
# Target
TARGET_DISTANCE = 15e3 # m
rtt = 2 * TARGET_DISTANCE / 3e8 # round-trip time [s]
if rtt < TIME_RESOL:
    print("Time resolution greater that rtt. Please fix.")
    raise ValueError
elif (rtt % TIME_RESOL) != 0:
    print("rtt must be mupltiple of time resolution. Please fix.")
    raise ValueError

# Q-learning settings.
EPISODES_LIMIT = 10_000 + 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = np.full(temp_buckets, 0.5)
EPS_DECAY = np.full(temp_buckets, 0.9998) # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1_000
Q_VALUE_MIN = -5
Q_VALUE_MAX = 0

class learning_agent:
    def __init__(self):
        self.freq = 0
    def __str__(self):
        return f"{self.time}, {self.freq}"
    def action(self, choice):
        if choice == 0: 
            self.freq = self.freq # keep same frequency
        if choice == 1: 
            self.freq += 10e6 # increase frequency
        if choice == 2:
            self.freq -= 10e6 # decrease frequency
        if self.freq < 0:
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
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
for temp_index in range(temp_buckets):
    ax1.plot(voltage_axis, tuning_law[temp_index,:])
ax1.set_title("VCO tuning voltage law")
ax1.set_ylabel("Frequency [Hz]")
ax1.set_xlabel("Voltage [V]")

# Let's define goal modulation, the one that must be learned from the learning agent
goal_modulation = 23e9 + (BANDWIDTH / CHIRP_PERIOD * time_axis)%3e9

# Plot goal modulation
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(time_axis, goal_modulation)
ax2.set_title("Goal modulation")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_xlabel("Time [s]")

# Q-table: random initialization
q_table = np.random.uniform(low=Q_VALUE_MIN, high=Q_VALUE_MAX, size=(ACTIONS_NUMBER,temp_buckets,time_buckets,freq_buckets))

# Q-learning process
rewards_history = []
VCO_history = np.zeros(time_buckets)
RX_history = np.zeros(time_buckets)
IF_history = np.zeros(time_buckets)

for episode in range(EPISODES_LIMIT):
    episode_reward = 0 # Reset this temporary value at every episode.
    current_temperature = getTemperature()
    time_bucket = 0
    VCO_output = learning_agent()
    VCO_history[time_bucket] = VCO_output.freq
    current_freq = round(abs(IF_history[time_bucket])/FREQ_RESOL)
    current_time = 0

    while time_bucket < (time_buckets-1):
        if np.random.random() > epsilon[current_temperature]: # exploiting action
            action = np.argmax(q_table[:, current_temperature, current_time, current_freq])
        else: # explorative action
            action = np.random.randint(0, ACTIONS_NUMBER)
        
        # Current state
        current_time = time_bucket
        current_freq = round(abs(IF_history[time_bucket])/FREQ_RESOL)
        
        # Take the action!
        VCO_output.action(action)
        # Evaluate reward for this time step.
        reward = - abs(IF_history[time_bucket])
        # Time has passed
        time_bucket += 1

        # New state
        VCO_history[time_bucket] = VCO_output.freq
        if (time_bucket*TIME_RESOL < rtt):
            RX_history[time_bucket] = 0
        else:
            RX_history[time_bucket] = VCO_history[time_bucket - 1]
        IF_history[time_bucket] = VCO_output.freq - RX_history[time_bucket]
        new_time = current_time + 1
        new_freq = round(abs(IF_history[time_bucket])/FREQ_RESOL)

        # Evaluate new Q-value
        max_future_q = np.max(q_table[:, current_temperature, new_time, new_freq])
        current_q = q_table[action, current_temperature, current_time, current_freq]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[action, current_temperature, current_time, current_freq] = new_q
        episode_reward += reward
    if episode % SHOW_EVERY == 0:
            print(f"On episode #{episode}, epsilon is {epsilon}")
            # Show how the training is going on
            fig3 = plt.figure(3)
            ax3 = fig3.add_subplot(1,1,1)
            # ax3.plot(time_axis, goal_modulation)
            ax3.set_title("Mixer signals")
            ax3.set_ylabel("Frequency [Hz]")
            ax3.set_xlabel("Time [s]")
            # ax3.plot(time_axis, VCO_history[:])
            # ax3.plot(time_axis, RX_history[:])
            ax3.plot(time_axis, IF_history[:])
            # fig4 = plt.figure(4)
            # ax4 = fig4.add_subplot(1,1,1)
            input("press enter...")
            plt.close(3)
    rewards_history.append(episode_reward)
    epsilon[current_temperature] *= EPS_DECAY[current_temperature]
moving_avg = np.convolve(rewards_history, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"Reward {SHOW_EVERY}ma")
# plt.xlabel("episode #")
# plt.show()

input("Press [Enter] to end the program...")
plt.close('all')