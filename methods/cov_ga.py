import numpy as np
import random
from deap import base, creator, tools, algorithms
import torch
def evaluate_individual(individual, X, target_covariance):
    indices = [i for i, included in enumerate(individual) if included]
    S = X[indices, :]
    covariance = S.T @ S
    return (np.linalg.norm(covariance - target_covariance, 'fro')/np.linalg.norm(target_covariance, 'fro'),)

def cov_ga(X, m, n_population=100, n_generations=40):
    target_covariance = X.T @ X
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[0])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, X=X, target_covariance=target_covariance)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=n_population)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

    best_ind = tools.selBest(population, 1)[0]
    best_indices = [i for i, included in enumerate(best_ind) if included]
    idxes = best_indices[:m]
    idxes = torch.tensor(idxes)
    return idxes
