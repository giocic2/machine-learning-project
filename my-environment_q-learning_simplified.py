import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()

# Resolutions
TEMP_RESOL = 10 # K
TIME_RESOL = 1 # s
IF_FREQ_RESOL = 1 # Hz
VOLTAGE_RESOL = 0.1 # V


# Q-table variables.
# Number of possible actions for the learning agent
ACTIONS_NUMBER = 5
print("Number of possible actions: ", ACTIONS_NUMBER)
# Time
START_TIME = 0 # seconds
END_TIME = 10 # seconds
time_buckets = round((END_TIME - START_TIME) / TIME_RESOL) + 1
print("Time buckets: ", "{:,}".format(time_buckets))
time_axis = np.linspace(start=START_TIME, stop=END_TIME, num=time_buckets, endpoint=True)
# Temperature
MIN_TEMP_C = 10 # °C
MAX_TEMP_C = 20 # °C
min_temp = MIN_TEMP_C + 273.15 # K
max_temp = MAX_TEMP_C + 273.15 # K
temp_buckets = round((max_temp - min_temp) / TEMP_RESOL) + 1
print("Temperature buckets: ", "{:,}".format(temp_buckets))
temp_axis = np.linspace(start=min_temp, stop=max_temp, num=temp_buckets, endpoint=True)
# Q-table size
print("Q-table size: ", "{:,}".format(ACTIONS_NUMBER * temp_buckets * time_buckets), end='\n\n')

# Environment variables.
# DAC
MIN_VOLTAGE = 0.5 # Volt
MAX_VOLTAGE = 5 # Volt
voltage_buckets = round((MAX_VOLTAGE - MIN_VOLTAGE) / VOLTAGE_RESOL) + 1
voltage_axis = np.linspace(start=MIN_VOLTAGE, stop=MAX_VOLTAGE, num=voltage_buckets, endpoint=True)
print("Voltage resolution: ", VOLTAGE_RESOL, " V")
# FMCW linear modulation (goal)
BANDWIDTH = 10 # Hz
CHIRP_PERIOD = END_TIME # s
START_FREQUENCY = 0 # Hz
# Target
target_IF = 0
print("Target IF: ", "{:,}".format(target_IF), " Hz")
# IF frequencies
MIN_IF_FREQ = 0 # Hz
MAX_IF_FREQ = target_IF * 2 # Hz
freq_buckets = round((MAX_IF_FREQ - MIN_IF_FREQ) / IF_FREQ_RESOL) + 1
freq_axis = np.linspace(start=MIN_IF_FREQ, stop=MAX_IF_FREQ, num=freq_buckets, endpoint=True)
print("IF frequency resolution: ", "{:,}".format(IF_FREQ_RESOL), " Hz")

# Q-learning settings.
EPISODES_LIMIT = 1000 + 1
SHOW_EVERY = 10
LEARNING_RATE = 0.5
DISCOUNT = 0.5
epsilon = np.full(temp_buckets, 0.5)
EPS_DECAY = np.full(temp_buckets, 0.99) # Every episode will be epsilon*EPS_DECAY
Q_VALUE_MIN = -5
Q_VALUE_MAX = 0

class learning_agent:
    def __init__(self):
        self.voltage_variation = 0
    def action(self, choice, previous_voltage, next_voltage):
        if choice == 0: 
            next_voltage = previous_voltage # keep same frequency
        if choice == 1: 
            next_voltage = previous_voltage + 1 * VOLTAGE_RESOL # increase DAC output voltage FINE
        if choice == 2:
            next_voltage = previous_voltage - 1 * VOLTAGE_RESOL # decrease DAC output voltage FINE
        if choice == 3:
            next_voltage = previous_voltage + 10 * VOLTAGE_RESOL # increase DAC output voltage COARSE
        if choice == 4:
            next_voltage = previous_voltage - 10 * VOLTAGE_RESOL # decrease DAC output voltage COARSE
        # Check if out of range
        if next_voltage < MIN_VOLTAGE:
            next_voltage = MIN_VOLTAGE
        elif next_voltage > MAX_VOLTAGE:
            next_voltage = MAX_VOLTAGE
        return next_voltage

def getTemperature():
    if np.random.random() > 0:
        temperature_bucket = 0
    else: 
        temperature_bucket = 1
    return temperature_bucket, MIN_TEMP_C + TEMP_RESOL * temperature_bucket

# Model of the VCO tuning law
tuning_law = np.ndarray((temp_buckets, voltage_buckets))
# VCO tuning law: coefficients
a1 = 0.1 # Hz, dynamic of the log function
a2 = 2 # Hz, offset for the log function
a3 = 0.01 # Hz/K, temperature dependence
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
goal_modulation = 23e9 + (BANDWIDTH / CHIRP_PERIOD * time_axis) % 3e9

# Plot goal modulation
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(time_axis, goal_modulation)
ax2.set_title("Goal modulation")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_xlabel("Time [s]")

# Q-table: random initialization
q_table = np.random.uniform(low=Q_VALUE_MIN, high=Q_VALUE_MAX, size=(ACTIONS_NUMBER,temp_buckets,time_buckets))

# Q-learning process
rewards_history = []
DAC_history = np.ones((temp_buckets, time_buckets)) * MIN_VOLTAGE # DAC signals that the system is learning.
DAC = learning_agent()
IF_history = np.zeros(time_buckets)
IF_deviation = np.zeros(time_buckets)
fig3 = plt.figure(3) # On-going results of the training

for episode in range(EPISODES_LIMIT):
    # Reset this temporary values at every episode.
    episode_reward = 0
    VCO_history = np.zeros(time_buckets)
    RX_history = np.zeros(time_buckets)
    IF_history = np.zeros(time_buckets)
    IF_deviation_OLD = IF_deviation
    IF_deviation = np.zeros(time_buckets)

    time_bucket = 0
    current_time_bucket = time_bucket
    [current_temperature_bucket, current_temperature] = getTemperature()
    current_voltage = DAC_history[current_temperature_bucket, current_time_bucket]

    VCO_output = a1 * np.log(current_voltage) + a2 - (a3 * (current_temperature - a4))
    VCO_history[time_bucket] = VCO_output

    while time_bucket < (time_buckets-1):
        if np.random.random() > epsilon[current_temperature_bucket]: # exploiting action
            action = np.argmax(q_table[:, current_temperature_bucket, current_time_bucket])
        else: # explorative action
            action = np.random.randint(0, ACTIONS_NUMBER)
        
        # Current state
        current_time_bucket = time_bucket
        IF_deviation[current_time_bucket] = round((IF_history[current_time_bucket] - target_IF)/IF_FREQ_RESOL)
        # if IF_history[current_time_bucket] > MAX_IF_FREQ:
        #     IF_history[current_time_bucket] = MAX_IF_FREQ
        # elif IF_history[current_time_bucket] < MIN_IF_FREQ:
        #     IF_history[current_time_bucket] = MIN_IF_FREQ
        
        # Take the action!
        new_voltage = DAC.action(action, DAC_history[current_temperature_bucket, current_time_bucket], DAC_history[current_temperature_bucket, current_time_bucket+1])
        time_bucket += 1 # Increment for the while loop

        # New state
        new_time_bucket = current_time_bucket + 1
        DAC_history[current_temperature_bucket, new_time_bucket] = new_voltage
        VCO_output = a1 * np.log(new_voltage) + a2 - (a3 * (current_temperature - a4))
        VCO_history[new_time_bucket] = VCO_output
        if (new_time_bucket*TIME_RESOL < rtt):
            RX_history[new_time_bucket] = 0
        else:
            RX_history[new_time_bucket] = VCO_history[new_time_bucket - 1]
        IF_history[new_time_bucket] = VCO_output - RX_history[new_time_bucket]
        # if IF_history[new_time_bucket] > MAX_IF_FREQ:
        #     IF_history[new_time_bucket] = MAX_IF_FREQ
        # elif IF_history[new_time_bucket] < MIN_IF_FREQ:
        #     IF_history[new_time_bucket] = MIN_IF_FREQ
        IF_deviation[new_time_bucket] = round((IF_history[new_time_bucket] - target_IF)/IF_FREQ_RESOL)
        
        # Evaluate reward for this time step.
        if IF_deviation[new_time_bucket] < IF_deviation_OLD[new_time_bucket]:
            reward = +2
        elif IF_deviation[new_time_bucket] >= IF_deviation_OLD[new_time_bucket]:
            reward = -2
        if IF_deviation[current_time_bucket] == IF_deviation[new_time_bucket]:
            reward = +1
        new_time_bucket = current_time_bucket + 1
        new_freq = round(IF_history[new_time_bucket]/IF_FREQ_RESOL)

        # Evaluate new Q-value
        max_future_q = np.max(q_table[:, current_temperature_bucket, new_time_bucket])
        current_q = q_table[action, current_temperature_bucket, current_time_bucket]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[action, current_temperature_bucket, current_time_bucket] = new_q
        episode_reward += reward
    if episode % SHOW_EVERY == 0:
            print(f"On episode #{episode}, epsilon is {epsilon}")
            # Show how the training is going on
            plt.clf()
            # TX and RX signals
            ax3 = fig3.add_subplot(3,1,1)
            ax3.set_title(f"Mixer input signals (episode #{episode})")
            ax3.set_ylabel("Frequency [Hz]")
            ax3.set_xlabel("Time [s]")
            ax3.plot(time_axis, goal_modulation)
            ax3.plot(time_axis, VCO_history[:])
            # ax3.plot(time_axis, RX_history[:])
            # IF signal
            ax4 = fig3.add_subplot(3,1,2)
            ax4.set_title("IF signal")
            ax4.set_ylabel("Frequency [Hz]")
            ax4.set_xlabel("Time [s]")
            ax4.plot(time_axis, IF_history[:])
            # Tuning voltage over time
            ax5 = fig3.add_subplot(3,1,3)
            ax5.set_title("Tuning voltage")
            ax5.set_ylabel("Voltage [V]")
            ax5.set_xlabel("Time [s]")
            ax5.plot(time_axis, DAC_history[current_temperature_bucket, :])
            plt.pause(2)
    rewards_history.append(episode_reward)
    epsilon[current_temperature_bucket] *= EPS_DECAY[current_temperature_bucket]
# Moving average of reward history
moving_avg = np.convolve(rewards_history, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
# Plot reward history
fig4 = plt.figure(4)
ax = fig4.add_subplot(1,1,1)
ax.set_title("Mixer input signals")
ax.set_ylabel(f"Reward {SHOW_EVERY}ma")
ax.set_xlabel("episode #")
ax.plot([i for i in range(len(moving_avg))], moving_avg)

input("Press [Enter] to end the program...")