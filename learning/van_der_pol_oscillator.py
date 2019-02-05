import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Order of the DEs is y and then dy
def derivatives(variables, t, parameters):
    x, y = variables
    F_0, m, omega, omega_0, beta = parameters
    #Return list of derivatives by time 
    # Be careful that the order of the return is the same as that of the variables as 
    # solving y will give x and solving the long bit gives y
    return [y, (F_0 / m) * np.sin(omega * t) - 2 * beta * omega_0 * y - pow(omega_0, 2) * x] 


# Initial conditions
x_0 = 1
y_0 = 0
initial_conditions = [x_0, y_0]

# Parameters
F_0 = 1
m = 1.0
omega = 2
omega_0 = 1.0
beta = 0.05 #1 / np.sqrt(2) - 0.001
# Package paramters in a list for the ODE solver
parameters = [F_0, m, omega, omega_0, beta]

# Time array
t_end = 100
time_step = 0.005
t = np.arange(0.0, t_end, time_step)

solution = odeint(derivatives, initial_conditions, t, args=(parameters,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot y as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, solution[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('x')
ax1.set_xlim(0.,)

# Plot x as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, solution[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('y')
ax2.set_xlim(0.,)

# Plot y vs x
ax3 = fig.add_subplot(313)
twopi = 2.0*np.pi
ax3.plot(solution[:,0]%twopi, solution[:,1], '.', ms=1)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_xlim(0., twopi)


plt.tight_layout()
plt.show()
