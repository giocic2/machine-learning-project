import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy

# Resolutions
TIME_RESOL = 100e-6  # s
FREQ_RESOL = 1e6  # Hz
VOLTAGE_RESOL = 1e-3  # V
TEMP_RESOL = 10  # K

# Q-table and environment variables.
# Number of possible actions for the learning agent
ACTIONS_NUMBER = 11
print("Number of possible actions: ", ACTIONS_NUMBER)
# Time
START_TIME = 0  # seconds
END_TIME = 5e-3  # seconds
time_buckets = round((END_TIME - START_TIME) / TIME_RESOL) + 1
print("Time buckets: ", time_buckets)
time_axis = np.linspace(start=START_TIME, stop=END_TIME, num=time_buckets, endpoint=True)
# Frequency
MIN_IF_FREQ = 0  # Hz
MAX_IF_FREQ = 120e6  # Hz
freq_buckets = round((MAX_IF_FREQ - MIN_IF_FREQ) / FREQ_RESOL) + 1
print("Frequency buckets: ", freq_buckets)
freq_axis = np.linspace(start=MIN_IF_FREQ, stop=MAX_IF_FREQ, num=freq_buckets, endpoint=True)
# Temperature
MIN_TEMP_C = 0
MAX_TEMP_C = 50
min_temp = MIN_TEMP_C + 273.15  # K
max_temp = MAX_TEMP_C + 273.15  # K
temp_buckets = round((max_temp - min_temp) / TEMP_RESOL) + 1
print("Temperature buckets: ", temp_buckets)
temp_axis = np.linspace(start=min_temp, stop=max_temp, num=temp_buckets, endpoint=True)
# Voltage
MIN_VOLTAGE = 0.5  # Volt
MAX_VOLTAGE = 5  # Volt
voltage_buckets = round((MAX_VOLTAGE - MIN_VOLTAGE) / VOLTAGE_RESOL) + 1
print("Voltage buckets: ", voltage_buckets)
voltage_axis = np.linspace(start=MIN_VOLTAGE, stop=MAX_VOLTAGE, num=voltage_buckets, endpoint=True)
# FMCW linear modulation
# BANDWIDTH = 3e9  # Hz
BANDWIDTH = 3e3 # Hz (test RL algorithm)
CHIRP_PERIOD = END_TIME  # s
START_FREQUENCY = 23e9  # Hz
# Target
TARGET_DISTANCE = 15e3  # m
rtt = 2 * TARGET_DISTANCE / 3e8  # round-trip time [s]
if rtt < TIME_RESOL:
    print("Time resolution greater that rtt. Please fix.")
    raise ValueError
elif (rtt % TIME_RESOL) != 0:
    print("rtt must be mupltiple of time resolution. Please fix.")
    raise ValueError
target_IF = (BANDWIDTH / CHIRP_PERIOD * rtt) / 1e9
print("Target IF: ", target_IF, " Hz")

# VCO tuning law: coefficients
# a1 = 1.969e9  # Hz, dynamic of the log function
a1 = 1.969e3 # Hz (test RL algorithm)
a2 = 23.947e9  # Hz, offset for the log function
a3 = 10e6  # Hz/K, temperature dependence
a4 = 233  # K, offset for the temperature drift

goal_modulation = 23e9 + (BANDWIDTH / CHIRP_PERIOD * time_axis) % 3e9


def tuning_law(voltage, temp):
    return (a1 * np.log(voltage) + a2 - (a3 * (temp - a4))) / 1e9


def get_temperature():
    if np.random.random() > 0:
        temperature_bucket = 0.0
    else:
        temperature_bucket = 1.0
    return temperature_bucket


class DACGym(gym.Env):
    def __init__(self):
        # gym mandatory members
        self.action_space = gym.spaces.Box(low=MIN_VOLTAGE, high=MAX_VOLTAGE, shape=(1,))
        self.reward_range = (-1, 1)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.step_number = 0
        self.max_steps = 100
        self.temp = 0.0
        self.last_vco_out = 0.0

    def step(self, action):
        voltage_in = float(action)

        vco_out = tuning_law(voltage_in, self.temp)
        if_out = vco_out - self.last_vco_out

        if_error = float(target_IF - if_out)
        if_array.append(if_error)
        step_obs = np.array([if_error, voltage_in, self.temp])
        self.last_vco_out = vco_out

        step_reward = np.clip(float(-abs(if_error)), -30, 0) / 30

        if self.step_number == self.max_steps:
            step_done = True
            self.step_number = 0
        else:
            step_done = False
            self.step_number += 1

        return step_obs, step_reward, step_done, {}

    def reset(self):
        self.temp = get_temperature()
        vco_out = tuning_law(MIN_VOLTAGE, self.temp)
        if_error = float(target_IF - vco_out)

        self.last_vco_out = vco_out

        obs = np.array([if_error, MIN_VOLTAGE, self.temp])
        return obs

    def render(self, mode="human"):
        return np.array([0.0, 0.0])


if __name__ == '__main__':
    if_array = []
    env = DACGym()
    policy_kwargs = dict(
        net_arch=dict(qf=[256, 128, 64], pi=[256, 128, 64])
    )

    model = SAC(policy=SACPolicy, env=env, verbose=1,
                policy_kwargs=policy_kwargs,
                device='cuda', train_freq=1, batch_size=256)
    model.learn(1_000_000, log_interval=10)
    model.save(f'last.zip')

    plt.plot(np.array(if_array))
    plt.show()
