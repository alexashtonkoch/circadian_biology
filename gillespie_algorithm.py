import numpy as np
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
    population_over_time = np.array([])
    time_points = np.array([])
    t = 0
    while t < t_stop:
        # 
        np.append(time_points, t)
        np.append(population_over_time, population.copy())
        reaction_index, time_interval = gillespie_draw(parameters, propensity_function, population)
        np.add(population, transitions[reaction_index]) # Check if np.add can be replaced by + operator 
        t += time_interval    
        print(reaction_index)
        '''
            REACTION INDEX IS ONLY ZERO
        '''
    print(population_over_time)
    print(time_points)
    return population_over_time, time_points


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
simulation_time = 1000

# Propensities needs to be a function as the number of molcules affects the liklihood of a reaction
def protein_production(parameters, population):
    beta_m, beta_p, gamma = parameters
    m, p = population
    return np.array([beta_m, m, beta_p * m, gamma * p])

#
population, time_points = gillespie_algorithm(protein_production_parameters, protein_production, protein_production_transitions, population, simulation_time)

print(time_points)
print(population)

'''
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot mRNA trajectories
ax[0].plot(time_points, population, '-', lw=0.3, alpha=0.2)

# Plot protein trajectories
ax[1].plot(time_points, population, 'k-', lw=0.3, alpha=0.2)

# Label axes
ax[0].set_xlabel('dimensionless time')
ax[1].set_xlabel('dimensionless time')
ax[0].set_ylabel('number of mRNAs')
ax[1].set_ylabel('number of proteins')
plt.tight_layout()
plt.show()
'''

             
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
