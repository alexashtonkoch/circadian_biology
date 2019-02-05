import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
 
# Order of the DEs is y and then dy
def derivatives(variables, t, parameters):
    x, y = variables
    #Return list of derivatives by time 
    # Be careful that the order of the return is the same as that of the variables as 
    # solving y will give x and solving the long bit gives y
    return [y, -1 * mu * (pow(x, 2) - 1) * y - x] 
 
# Initial conditions
x_0 = 0.5
y_0 = 0
initial_conditions = [x_0, y_0]
 
# Parameters
mu = 1.5
# Package paramters in a list for the ODE solver
parameters = []
 
# Time array
t_end = 100
time_step = 0.0005
t = np.arange(0.0, t_end, time_step)
 
solution = odeint(derivatives, initial_conditions, t, args=(parameters,))
 
# Plot results
fig_1 = plt.figure(1, figsize=(8, 4), dpi = 600)
 
# Plot x as a function of time
ax1 = fig_1.add_subplot(311)
ax1.plot(t, solution[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('x')
ax1.set_xlim(0.,)
 
fig_2 = plt.figure(2, figsize=(8, 16), dpi = 600)

# Plot y vs x
ax2 = fig_2.add_subplot(311)
ax2.plot(solution[:,0], solution[:,1], '.', ms=1)
ax2.set_xlabel('x')
ax2.set_ylabel('dx/dt')
#ax3.set_xlim(-2, 2)
#ax3.set_ylim(-2, 2)

#x_0 = -2.0
#y_0 = 3.0
#initial_conditions = [x_0, y_0]
mu = 0.01
solution = odeint(derivatives, initial_conditions, t, args=(parameters,))

# Plot y vs x
ax3 = fig_2.add_subplot(312)
ax3.plot(solution[:,0], solution[:,1], '.', ms=1)
ax3.set_xlabel('x')
ax3.set_ylabel('dx/dt')
#ax3.set_xlim(-2, 2)
#ax3.set_ylim(-2, 2)
 
plt.tight_layout()
plt.show()
#fig_1.savefig('van_der_pol_oscillator_x.png', dpi = 600)
#fig_2.savefig('van_der_pol_oscillator_x_y.png', dpi = 600)
