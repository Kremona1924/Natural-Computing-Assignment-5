import numpy as np
import string

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

def initialise_population(pop_size, string_length):
    return [get_random_letter(string_length) for i in range(pop_size)]

def tournament_selection(pop, tournament_size, num):
    winners = []
    for _ in range(num):
        selection = np.random.choice(len(pop), size=tournament_size, replace=False) # Indices of individiuals in the tournament
        winner = selection[np.argmax([fitness(pop[i], target_string) for i in selection])] # Find index of fittest individual
        winners.append(pop[winner]) # Add fittest individual to output
    return winners

def create_new_population(pop):
    new_pop = []
    for i in range(len(pop)//2):
        parents = tournament_selection(pop, K, 2)
        offspring = crossover(*parents)
        offspring = [mutate(x, mu) for x in offspring]
        new_pop += offspring
    return new_pop

target_string = "CharlesDarwin"
L = len(target_string)

alphabet = np.array([letter for letter in string.ascii_letters])

K = 10 # Number of individuals per tournament
mu = 0.01 # Mutation rate
num_generations = 100
pop_size = 200

pop = initialise_population(pop_size, L)
for i in range(num_generations):
    pop = create_new_population(pop)
    if any([fitness(x, target_string) == len(target_string) for x in pop]):
        print("Target string found in generation ", i)
        break
