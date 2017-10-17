import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')

def test_fun(y, t):
    g = 9.81

    dydt = np.zeros_like(y)

    dydt[0] = y[2]
    dydt[1] = y[3]
    dydt[2] = -g + y[0]*y[3]**2 
    dydt[3] = -2*y[2]*y[3]/y[0]

    return dydt

Y0 = [10, 0, 0, 2]

t = np.linspace(0, 10, 1000)

res = scipy.integrate.odeint(test_fun, Y0, t)

x = res[:,0] * np.sin(res[:,1])
y = res[:,0] * np.cos(res[:,1])

plt.figure()
plt.plot(x, y)
plt.show()