import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]) 
y = np.dot(X, np.array([1, 2])) + 3 # y = 1 * x_0 + 2 * x_1 + 3
plt.plot(X)
plt.plot(y)
plt.show()
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(np.array([[3, 5]])))

# Test: VCO freq vs voltage
a0 = 1
a1 = 2
a3 = 0.7
voltage = np.linspace(start=0, stop=5, num=21, endpoint=True) # V
freqVCO = a0 + a1*voltage + a3*(voltage**2) # Hz
plt.plot(voltage, freqVCO)
plt.xlabel('voltage (V)')
plt.ylabel('frequency (Hz)')
plt.show()
voltage_samples = np.asarray([0.1, 1, 0.7, 0.3, 2, 4, 3.5, 5])
freqVCO_samples = a0 + a1*voltage_samples + a3*(voltage_samples**2)
voltage_feature = np.column_stack((voltage_samples, voltage_samples**2))
plt.plot(voltage_samples)
plt.plot(freqVCO_samples)
plt.show()
reg = LinearRegression().fit(voltage_feature, freqVCO_samples)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(np.array([6, 6**2])))