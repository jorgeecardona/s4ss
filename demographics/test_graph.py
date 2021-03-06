import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

class BaseContainer:
    """
    Container
    =========

    First container on the simulation.
    
    """
  
    def __init__(self):

        # Create a graph to hold the objects.
        self._graph = nx.DiGraph()

    def add_node(self, node):

        # add node.
        self._graph.add_node(node)
        self._graph.add_edge(self, node)
        
        # If node has a graph combine them.
        if hasattr(node, '_graph'):
            self._graph = nx.compose(self._graph, node._graph)                            

        # The node needs a reference to the graph.
        node.set_graph(self._graph)

    def step(self, context={}):        
        [n.step(context=context) for n in self._graph.successors(self)]
                        

class BaseAgent(object):
    """
    Agent
    =====

    An agent should have some ways to compute some valuable data that goes to its
    environment as interactions with other agents, and finally to final result of
    some simulation.
    
    Is hard to define what can be a model, but we can try to see some examples of
    what can be a model:

    - Human.

    - Clustering of humans as companies, home, countries.
    
      - We need a mechanism to create the clustering and allow an agent to belong
        to several clusters.

    - Environment (earth, we can model the biological capacity of a region)


    Clustering
    ----------

    A cluster is another agent with a list of composing agents, and each inner agent have
    a way to lookup its clustering agents. This seems easier to model with some graph.

    The clustering is meaninful in some context, this context can track the tags,
    eg: homes, companies are important in the context of the country, countries
    in the overall society, so, a clustering can serve also the purpose of a container.
    

    """

    def set_graph(self, graph):
        self._graph = graph


class Society(BaseAgent):

    def __init__(self, initial_population=100):
        # Create a directed graph to keep the humans in place.
        self._graph = nx.DiGraph()
        self._graph.add_node(self)

        # Add a fixed number of humans.
        self.add_humans_in_batch(initial_population)

        # Set ages randomly.
        for human in self._graph.successors(self):
            human.age = np.random.randint(0, 80)

    def add_humans_in_batch(self, n=1000):
        females = np.random.binomial(n, 0.5)        
        [self.add_human(Human('female')) for i in range(females)]
        [self.add_human(Human('male')) for i in range(n - females)]        
                
    def add_human(self, h):
        self._graph.add_edge(self, h)

    def remove_human(self, h):
        self._graph.remove_edge(self, h)
        h.kill()
               
    def step(self, context={}):
        # A single step will do a lot of things: birth-death, ...

        # Get everyone a bit older:
        [h.step() for h in self._graph.successors(self)]
        
        # Get birth-death rates by context.
        birth_rate = context.get('birth-rate')
        death_structure = context.get('death-structure')
                
        # Compute the increase.
        population_inc = np.random.binomial(self.population, birth_rate)  # TODO: use population in reproductive age only

        # Add new humans, half and half.
        self.add_humans_in_batch(population_inc)
                
        # TODO: This should be somthing easy to ask in some graph tool.
        total_dec = 0
        for range in death_structure:
            age_min = range[0]
            age_max = range[1]
            pr_death = range[2]
            pop = [h for h in self._graph.successors(self) if h.age >= age_min and h.age <= age_max]
        
            # Compute the decrease.
            population_dec = np.random.binomial(len(pop), pr_death)
            total_dec += population_dec

            # Pick randomly humans.
            if population_dec > 0:
                deceases = np.random.choice(pop, population_dec, replace=False)
                [self.remove_human(h) for h in deceases]
        
        print("population: %6d, offset: %3d births - %3d deaths = %3d" % (self.population, population_inc, total_dec, population_inc - total_dec ))

    # TODO: do we want to combine the 'admin' property with 'data-simulation' properties?
    @property
    def population(self):
        return len(self._graph.successors(self))       

    @property
    def population_distribution(self):
        ages = [h.age for h in self._graph.successors(self)]
        if len(ages) > 0:
            return [(i, ages.count(i)) for i in range(max(ages) + 1)]
        return []
                
class Human(BaseAgent):

    _corpses = []

    def __new__(cls, *args, **kwargs):
        # Reuse a corpse.
        if len(cls._corpses) > 0:
            h = cls._corpses.pop()
            return h
        return object.__new__(Human, *args, **kwargs)
            
    def __init__(self, gender):
        # Human attributes.
        self.age = 0
        self.gender = gender
        
    def step(self):
        self.age += 1                

    def kill(self):
        self.age = -1
        self._corpses.append(self)        
        
if __name__ == "__main__":

    # read context from file
    context = {}
    try:
        with open('society.context', 'r') as f:
            s = f.read()
            context = ast.literal_eval(s)
    except:
        pass

    for i in range(1):
    
        # Create container.
        c = BaseContainer()

        s = Society(initial_population=1000)
    
        c.add_node(s)

        # Record values.
        population = []
        
        for j in range(500):
            population.append(s.population)
            if context:
                c.step(context)
            else:
                c.step()
            
        print('Run %2d: Final population: %6d' % (i + 1, population[-1]))

        # Plot population
        plt.subplot(211)
        plt.plot(population)
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()

        # Plot distribution.
        plt.subplot(212)
        plt.plot(*zip(*s.population_distribution))
        plt.xlabel('Age')
        plt.ylabel('Population')
        plt.grid()        

#    plt.show()
    
