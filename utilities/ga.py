import random
from deap import creator, base, tools, algorithms
import warnings
import numpy as np

warnings.filterwarnings("default")


# https://deap.readthedocs.io/en/master/api/tools.html

class GASearchComb(object):

    def __init__(self, n_individual, n_selected, n_population=2000, tournsize=None, cxpb=.5, mutpb=.2, indpb=.05, ngen=100, random_state=1985):

        random.seed(random_state)
        self.n_individual = n_individual
        self.n_selected = n_selected
        self.n_population = n_population
        self.tournsize = tournsize
        if (self.tournsize == None):
            self.tournsize = int(self.n_population * .01)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.ngen = ngen
        creator.create("FitnessMax", base.Fitness, weights=(1.0,), )
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_gen", random.randint, 0, n_individual - 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_gen, n=self.n_selected)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        self.population = self.toolbox.population(n=self.n_population)
        return

    def evaluate(self, individual):
        raise Exception('no inheritance')

    def run(self, ntop=1):
        for gen in range(self.ngen):
            offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            self.population = self.toolbox.select(offspring, k=len(self.population))
        return tools.selBest(self.population, k=ntop)

    # def _evalute_n_population(self, n_component, n_features, init_n_population=100, feature_adoption_rate=.9):
    #     n_population = init_n_population
    #     selected_n_feature = 0
    #     while selected_n_feature / n_features < feature_adoption_rate:
    #         n_population = int(n_population * 1.2)
    #         k = []
    #         for j in np.arange(n_population):
    #             k.append([random.randint(0, n_features - 1) for i in np.arange(n_component)])
    #         k = np.array(k)
    #         selected_n_feature = np.unique(k).shape[0]
    #     return n_population
