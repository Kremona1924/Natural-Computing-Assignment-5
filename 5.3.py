import numpy as np
import string
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def get_random_letter(number = None):
    return np.random.choice(alphabet, size=number, replace=True)

def crossover(x, y):
    crossover_point = np.random.randint(1, len(x) - 1)
    return np.concatenate([x[:crossover_point], y[crossover_point:]]), np.concatenate([y[:crossover_point], x[crossover_point:]])

def mutate(x, mutation_rate):
    for i in range(len(x)):
        if np.random.rand() < mutation_rate:
            x[i] = get_random_letter()
    return x

def fitness(x, target_string):
    fitness = 0
    for letter, target in zip(x, target_string):
        if letter == target:
            fitness += 1
    return fitness

def hamming_distance(x, y):
    distance = 0
    for a, b in zip(x, y):
        if a != b:
            distance += 1
    return distance

def compute_diversity(pop):
    # Since we use the full population, this function takes a while. On my machine, its still not too slow, but the population could be sampled to improve runtime if needed
    avg_distance = 0
    for x in pop:
        for y in pop:
            avg_distance += hamming_distance(x,y)
    avg_distance /= len(pop)**2
    return avg_distance 

def initialise_population(pop_size, string_length):
    return [get_random_letter(string_length) for i in range(pop_size)]

def tournament_selection(pop, tournament_size, num):
    winners = []
    for _ in range(num):
        selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
        winner = selection[np.argmax([fitness(pop[i], target_string) for i in selection])] # Find index of fittest individual
        winners.append(pop[winner]) # Add fittest individual to output
    return winners

def create_new_population(pop, mu, k):
    new_pop = []
    for i in range(len(pop)//2):
        parents = tournament_selection(pop, k, 2)
        offspring = crossover(*parents)
        offspring = [mutate(x, mu) for x in offspring]
        new_pop += offspring
    return new_pop


### Hyperparameters etc.
target_string = "CharlesDarwin"
L = len(target_string)
alphabet = np.array([letter for letter in string.ascii_letters])

K = 2 # Number of individuals per tournament
mu = 1/L # Mutation rate

num_generations = 100
pop_size = 200
iterations = 10

diversity_measurement_interval = 10
finish_time = []

# Main experiments
def vary_mu():
    # Vary mutation rate and record finish time
    K = 2
    mus = [0, 1/L, 3/L]
    mus_labels = ["μ = 0", "μ = 1/L", "μ = 3/L"]
    df = pd.DataFrame()
    for m, m_label in zip(mus, mus_labels):
        mu = m
        finish_times = []
        for _ in range(iterations):
            pop = initialise_population(pop_size, L)
            found_target = False
            for i in range(num_generations):
                pop = create_new_population(pop, mu, K)
                if any([fitness(x, target_string) == len(target_string) for x in pop]):
                    finish_times.append(i)
                    found_target = True
                    break
            if not found_target: finish_times.append(num_generations)
        df[m_label] = finish_times
    sns.swarmplot(data=df)
    plt.ylabel("Generation")
    plt.show()
    


def measure_diversity():
    # Measure population diversity for several values of mu
    K = 2
    mus = [0, 1/L, 3/L]
    for m in mus:
        mu = m
        diversity = np.empty((iterations,10))
        for iter in range(iterations):
            pop = initialise_population(pop_size, L)
            current_diversity = []
            for i in range(num_generations):
                if (i % diversity_measurement_interval == 0):
                    current_diversity.append(compute_diversity(pop))
                pop = create_new_population(pop, mu, K)
            diversity[iter] = current_diversity
        mean_diversity = np.mean(diversity, axis=0)
        plt.plot(np.arange(len(mean_diversity))*diversity_measurement_interval, mean_diversity)

    plt.xlabel("Generation")
    plt.ylabel("Mean hamming distance")
    plt.legend(["μ = 0", "μ = 1/L", "μ = 3/L"])
    plt.show()

vary_mu()
measure_diversity()
