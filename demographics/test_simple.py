import numpy as np
import matplotlib.pyplot as plt
import ast

class Demographics:
    # Assume a global initial distribution.
    # http://www.indexmundi.com/world/age_structure.html
    def __init__(self, 
                 initial_population=10000, 
                 age_structure=[(0, 14, 0.2579), (15, 24, 0.1661), (25, 54, 0.4078), (55, 64, 0.0851), (65, 90, 0.0832)]):

        self.max_age = 120
        self.min_age_to_reproduce = 12
        self.density = self.generate(initial_population, age_structure)

    def generate(self, n, structure):

        # Create a density.
        density = [0] * self.max_age

        for section in structure:

            section_size = section[1] - section[0] + 1

            for age in range(section[0], section[1] + 1):
                density[age] = np.random.binomial(n / section_size, section[2])

        return density

    def step(self, facts):
        # Default data from: http://www.indexmundi.com/world/birth_rate.html and http://www.indexmundi.com/world/death_rate.html

        # Deaths.
        for section in facts['death-structure']:
            for age in range(section[0], min(section[1] + 1, self.max_age)):
                if self.density[age] > 0:
                    self.density[age] = max(0, self.density[age] - np.random.binomial(self.density[age], section[2]))

        # Increase everyone's age.
        self.density = [0] + self.density[:(self.max_age-1)] # all individuals at max_age
                    
        # New borns.
        if sum(self.density[:self.min_age_to_reproduce]) > 0:
            self.density[0] += np.random.binomial(sum(self.density), facts['birth-rate'])


    def run(self, n=200, facts={'birth-rate': 18.7 / 1000, 'death-structure': [(0, 0, 7.09 / 1000), (1, 14, 0.425 / 1000), (15, 24, 0.75 / 1000), (25, 54, 7.04 / 1000), (55, 64, 8.87 / 1000), (65, 74, 20.28 / 1000), (75, 84, 51.75 / 1000), (85, 200, 205.0 / 1000)]}):
        # Death structure taken from: https://www.census.gov/compendia/statab/2012/tables/12s0110.pdf

        for i in range(n):
            self.step(facts=facts)

if __name__ == "__main__":

    # read context from file
    context = {}
    try:
        with open('society.context', 'r') as f:
            s = f.read()
            context = ast.literal_eval(s)
    except:
        pass

    n = 200
    for i in range(10):
        # Create a Demographics
        d = Demographics()
        if context:
            d.run(n, context)
        else:
            d.run(n)
        plt.plot(d.density)
        plt.xlabel('Age')
        plt.ylabel('Population size')

    plt.grid()
    plt.show()

