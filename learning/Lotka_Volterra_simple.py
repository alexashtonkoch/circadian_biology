import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

'''
    dx/dt = ax - bxy
    dy/dt = cxy - dy
    
    x is the number of prey 
    y is the number if predator
'''

def derivatives(variables, t, parameters):
    x, y = variables
    a, b, c, d = parameters
    #Return list of derivatives by time 
    # Be careful that the order of the return is the same as that of the variables as 
    # solving y will give x and solving the long bit gives y
    return [a* x - b * x * y, c * x * y - d * y] 

# Initial conditions
x_0 = 1.1
y_0 = 1.1
initial_conditions = [x_0, y_0]

# Parameters
a = 0.6
b = 1.3
c = 1
d = 1
# Package paramters in a list for the ODE solver
parameters = [a, b, c, d]

# Time array
t_end = 50
time_step = 0.0005
t = np.arange(0.0, t_end, time_step)

solution = odeint(derivatives, initial_conditions, t, args=(parameters,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot x as a function of time
ax1 = fig.add_subplot(211)
ax1.plot(t, solution[:,0], label="prey")
ax1.plot(t, solution[:,1], label="predator")
ax1.set_xlabel('time')
ax1.set_ylabel('Number')
ax1.set_xlim(0.,)
ax1.legend()

# Plot y vs x
ax2 = fig.add_subplot(212)
ax2.plot(solution[:,0], solution[:,1], '.', ms=1)
ax2.set_xlabel('Prey')
ax2.set_ylabel('Predator')
ax2.set_aspect('equal', 'datalim')
ax2.set_xlim(0.,)

plt.tight_layout()
plt.show()
