import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BirthDeathProcess:
    """
    Birth death process
    ===================

    Given a population of 'n', we model the probability of having a child in the time
    step interval as a bernoulli trial for each person, then the sum gives a multinomial
    distribution.

    In the limit (n goes to inf) a multinomial approach a normal (central limit theorem)
    with mean np and var np(1 - p), numpy switch to normal when n * p > 5:

    http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html    
    
    """
    switch_to_normal_limit = 1000

        
    def __init__(self, birth_rate=1, death_rate=1, population=0):

        # Store initial population.
        self.initial_population = population
        
        # A probability in between [0,1]:
        self.birth_rate = birth_rate
        self.death_rate = death_rate

        # Reset the process.
        self.reset()

    def reset(self):
        # NOTE: Should we encapsulate the variables inside a dict or a list?
        self.population = self.initial_population
                    
    def step(self):

        # Compute the increase.
        population_inc = np.random.binomial(self.population, self.birth_rate)

        # Compute the decrease.
        population_dec = np.random.binomial(self.population, self.death_rate)        

        # Update the size.
        self.population += population_inc - population_dec
        self.population = max(0, self.population)
        if self.population < 2: # population collapse, assuming sexual reproduction
          self.birth_rate = 0
          self.deadth_rate = 0

        # Return some state.
        return (self.population, population_inc, population_dec)

class Container:

    def __init__(self):
        self.processes = {}
        self.monitor = {}

    def add_process(self, name, process, monitor=[]):

        # Save the process.
        self.processes[name] = process

        # save the variables to watch.
        for m in monitor:
            self.monitor['%s.%s' % (name, m)] = (process, m)

    def reset(self):
        
        # Reset all the processes in the container.
        for p in self.processes.values():
            p.reset()                
        
    def run(self, n=1000, monitor=[]):

        # Save data here and later pass it to a dataframe.
        variables = dict([(k, []) for k in self.monitor.keys()])
                        
        # TODO: Add some reset.        
        for i in range(n):

            # Store the data.
            [variables[k].append(getattr(v[0], v[1])) for k, v in self.monitor.items()]

            # Run a step.
            [p.step() for p in self.processes.values()]

        # Last data.
        [variables[k].append(getattr(v[0], v[1])) for k, v in self.monitor.items()]

        # Dataframe to hold the variables.        
        df = pd.DataFrame(columns=self.monitor.keys())

        # Pass the data to the dataframe (df are slow for per-row inserts)
        for k in self.monitor:
            df[k] = variables[k]
                
        return df

if __name__ == "__main__":
    # Create container.
    c = Container()    
    p = BirthDeathProcess(0.001, 0.00075, 16)

    # Add process.
    c.add_process('birth_death_process', p, monitor=['population'])

    # Run.
    x = 'birth_death_process.population';
    for i in range(16):
        c.reset()
        d = c.run(10000)
        print('Run %2d: Final population: %6d' % (i+1, d[x].tail(1)))
        d[x].plot()
    plt.show()

