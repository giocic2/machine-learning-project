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