import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Resolutions
TIME_RESOL = 100e-6 # s
FREQ_RESOL = 1e6 # Hz
VOLTAGE_RESOL = 1e-5 # V
TEMP_RESOL = 15 # K

# Temperature
MIN_TEMP_C = 0
MAX_TEMP_C = 50
min_temp = MIN_TEMP_C + 273.15 # K
max_temp = MAX_TEMP_C + 273.15 # K
temp_buckets = round((max_temp - min_temp) / TEMP_RESOL) + 1
print("Temperature buckets: ", temp_buckets)
temp_axis = np.linspace(start=min_temp, stop=max_temp, num=temp_buckets, endpoint=True)
# Voltage
MIN_VOLTAGE = 0.5 # Volt
MAX_VOLTAGE = 5 # Volt
voltage_buckets = round((MAX_VOLTAGE - MIN_VOLTAGE) / VOLTAGE_RESOL) + 1
print("Voltage buckets: ", voltage_buckets)
voltage_axis = np.linspace(start=MIN_VOLTAGE, stop=MAX_VOLTAGE, num=voltage_buckets, endpoint=True)

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
input("Press [Enter] to end the program...")