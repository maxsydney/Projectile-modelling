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
    dydt[2] = 0
    dydt[3] = -g

    return dydt

Y0 = [0, 10, 20, 0]

t = np.linspace(0, 10, 1000)

res = scipy.integrate.odeint(test_fun, Y0, t)


plt.figure()
plt.plot(res[:,0], res[:,1])
plt.show()