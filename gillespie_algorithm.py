import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

random.seed()

# Decides which reaction to use
def sample_reaction(propensity_distribution):
    sample = random.uniform(0,1)
    i = 0
    prob_sum = 0.0
    while prob_sum < sample:
        prob_sum += propensity_distribution[i]
        i += 1
    return i -1
          
def gillespie_draw(parameters, propensity_function, population):
    # Draw a reaction and the time it took to do the reaction
    propensities = propensity_function(parameters, population)
    
    propensity_sum = np.sum(propensities)
    
    # Calculatie the time interval between reactions
    time_interval = - (1 / propensity_sum) * np.log(random.uniform(0,1)) 
    
    propensity_distribution = propensities / propensity_sum # normalise propensities 
    
    
    # Decide which reaction happens by randomly sampling the discrete (normalised) propensity distribution
    reaction = sample_reaction(propensity_distribution)
    
    return reaction, time_interval
  
def gillespie_algorithm(parameters, propensity_function, transitions, population, t_stop):
    '''
    
    '''
    time_points = []
    population_over_time = []
    t = 0
    while t < t_stop:
        # 
        time_points.append(t)
        population_over_time.append(population)
        
        reaction_index, time_interval = gillespie_draw(parameters, propensity_function, population)
        op = transitions[reaction_index]
        
        t += time_interval
        population = np.add(population, op) # Check if np.add can be replaced by + operator   
    return np.array(time_points), np.array(population_over_time)


'''
  Reactions : propensities
  1. m --> m + 1 : beta_m
  2. m --> m - 1 : m
  3. p --> p + 1 : beta_p * m
  4. p --> p - 1 : gamma * p
  
  Initialise system in state m = p = 0
'''
  
protein_production_transitions = np.array([[1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]], dtype = np.int)
beta_m = 10
beta_p = 10
gamma = 0.4
protein_production_parameters = [beta_m, beta_p, gamma]

population = np.array([0, 0]) # Initialise the system
time_points = np.array([])
simulation_time = 100

# Propensities needs to be a function as the number of molcules affects the liklihood of a reaction
def protein_production(parameters, pop):
    beta_m, beta_p, gamma = parameters
    m, p = pop
    return np.array([beta_m, m, beta_p * m, gamma * p])

# Perform stochatsic simulations
simulations = 100
data = []
for i in range(simulations):
    random.seed()
    time_points, population = gillespie_algorithm(protein_production_parameters, protein_production, protein_production_transitions, population, simulation_time)
    data.append([time_points, population])
    # Re-initialise the system
    population = np.array([0, 0]) 
    time_points = np.array([])

data = np.array(data)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot mRNA trajectories
for i in range(simulations):
    ax[0].plot(data[i,0], ((data[i])[1])[:,0], '-')
    
# Plot mRNA mean
#ax[0].plot(data[0,0], (data[:,1]).mean(), '-')

'''

Mean currently doesn't work as the number of time points do not match and therefore cannot average over each point for the populations

'''

# Plot protein trajectories
for i in range(simulations):
    ax[1].plot(data[i,0], (data[i,1])[:,1], 'k-')

# Plot protein mean
#ax[1].plot(data, data[:,:,1].mean(axis=0), 'r-')

# Label axes
ax[0].set_xlabel('dimensionless time')
ax[1].set_xlabel('dimensionless time')
ax[0].set_ylabel('number of mRNAs')
ax[1].set_ylabel('number of proteins')
plt.tight_layout()
plt.show()


             
# Initialise the system in its initial state n=n(0)

# Figure out the states that can be accessed, m_1, ... m_M, which the system can 
# transition to from the state n(t)

# Set lambda = sum(transition probabilities from n to m_i)

# Draw a random number from an eponential distribution with parameter lambda 
# This is achived by setting tau = -(1/lambda) * ln(r) where r is a random number 
# over the interval (0,1]
# Now increment time by tau

# Execute one of the m_i possibilities event i=1,..,M from item 2. Select event i 
# with probability W_i/lambda and change the state of the system accordingly

# goto 2
