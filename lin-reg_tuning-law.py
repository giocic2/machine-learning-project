from random import sample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Temperature
MIN_TEMP_C = 0
MAX_TEMP_C = 50
min_temp = MIN_TEMP_C + 273.15 # K
max_temp = MAX_TEMP_C + 273.15 # K
# Voltage
MIN_VOLTAGE = 0.5 # Volt
MAX_VOLTAGE = 5 # Volt

# Samples of voltages and temperatures
SAMPLES_NUMBER = 1_000
samples = np.column_stack((MIN_VOLTAGE + np.random.rand(SAMPLES_NUMBER)*(MAX_VOLTAGE - MIN_VOLTAGE), np.random.normal(loc=25, scale=1, size=SAMPLES_NUMBER))) # [V, K]
plt.scatter(samples[:,0], samples[:,1])
plt.title("Voltage-temperature pairs")
plt.xlabel("voltage (V)")
plt.ylabel("temperature (C)")
plt.show()

# Model of the VCO tuning law
# VCO tuning law: coefficients
a1 = 1.969e9 # Hz, dynamic of the log function
a2 = 23.947e9 # Hz, offset for the log function
a3 = 10e6 # Hz/K, temperature dependence
a4 = 233 # K, offset for the temperature drift
VCOfreq_samples = np.zeros(SAMPLES_NUMBER)
samples[:,1] += 273.15 # From Celsius degree to Kelvin
for index in range(SAMPLES_NUMBER):
    VCOfreq_samples[index] = a1 * np.log(samples[index,0]) + a2 - (a3 * (samples[index,1] - a4))
    print(f"Sample #{index+1:}: {samples[index,1]-273.15:.1f} °C, {samples[index,0]:.2f} V, {VCOfreq_samples[index]/1e9:.2f} GHz")

voltage_feature = samples[:,0]
temp_feature = samples[:,1]+273.15
features = np.column_stack((temp_feature, voltage_feature, voltage_feature**2))

reg = LinearRegression().fit(features, VCOfreq_samples)
print(reg.score(features, VCOfreq_samples))
print(reg.coef_)
print(reg.intercept_)