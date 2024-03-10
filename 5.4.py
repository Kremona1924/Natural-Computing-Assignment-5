import numpy as np
import matplotlib.pyplot as plt
import random

# Load TSP data from file
def load_tsp_data(filename):
    with open(filename, 'r') as f:
        data = [list(map(float, line.split())) for line in f]
    return np.array(data)

# Calculate the total distance of a tour
def compute_distance_matrix(points):
    distance_matrix = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :]) ** 2).sum(axis=2))
    return distance_matrix

def calculate_total_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i-1], tour[i]] for i in range(len(tour)))

# Calculate the fitness score of a tour
def calculate_fitness(tour, distance_matrix):
    total_distance = calculate_total_distance(tour, distance_matrix)
    fitness_score = 1.0 / total_distance
    return fitness_score

# Generate the initial population of random tours
def generate_initial_population(population_size, number_of_cities):
    return [random.sample(range(number_of_cities), number_of_cities) for route in range(population_size)]

def tournament_selection(population, distance_matrix, tournament_size, num_winners):
    winners = []
    for _ in range(num_winners):
        selection = np.random.choice(len(population), size=tournament_size, replace=False) 
        winner = selection[np.argmax([calculate_fitness(population[i], distance_matrix) for i in selection])] 
        winners.append(population[winner])  # Add fittest individual to output
    return winners

def create_parent_pairs(winners):
    parent_pairs = []
    used_indices = set()  # To track which indices have been used

    while len(parent_pairs) < len(winners) // 2:
        # Randomly select two from the list of winners
        idx1, idx2 = np.random.choice(len(winners), size=2, replace=False)
        while idx1 in used_indices or idx2 in used_indices or idx1 == idx2:
            idx1, idx2 = np.random.choice(len(winners), size=2, replace=False)

        # Add the selected pair to the list of parent pairs
        parent_pairs.append((winners[idx1], winners[idx2]))
        # Mark these indices as used
        used_indices.add(idx1)
        used_indices.add(idx2)

    return parent_pairs

def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    # 1. Choose two random indices
    start, end = sorted(random.sample(range(size), 2))
    
    # 2. Keep the middle part
    child[start:end] = parent1[start:end]
    
    # 3 en 4. Fill the remaining positions with the remaining numbers from the second parent
    parent2_pointer = end
    child_pointer = end
    while -1 in child:
        if parent2[parent2_pointer % size] not in child:
            child[child_pointer % size] = parent2[parent2_pointer % size]
            child_pointer += 1
        parent2_pointer += 1
    
    return child

def swap_mutation(tour):
    a, b = random.sample(range(len(tour)), 2)
    tour[a], tour[b] = tour[b], tour[a]
    return tour

# keep the best 2 individuals from the population
def elitism(population, distance_matrix, elite_count):
    return sorted(population, key=lambda ind: calculate_fitness(ind, distance_matrix), reverse=True)[:elite_count]

def run_genetic_algorithm(filename, population_size, generations, mutation_rate, tournament_size, elite_count=2):
    tsp_data = load_tsp_data(filename)
    distance_matrix = compute_distance_matrix(tsp_data)
    population = generate_initial_population(population_size, len(tsp_data))

    # print one of the initial routes
    print(population[0])

    best_solution_overall = None
    best_distance_overall = float('inf')

    for generation in range(generations):
        # Grab the elite individuals from the population
        elite = elitism(population, distance_matrix, elite_count)

        # The elite can also be used as parents for the next generation
        # We only need to select (population_size - elite_count) additional parents
        winners = tournament_selection(population, distance_matrix, tournament_size, population_size - elite_count)

        parent_pairs = create_parent_pairs(winners)

        # Cross over the parents to create children and apply mutation
        children = []
        for parent1, parent2 in parent_pairs:
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)
            children.extend([child1, child2])

        for child in children:
            if random.random() < mutation_rate:
                child = swap_mutation(child)

        # add the elite to the children AFTER crossover and mutation
        children.extend(elite)

        population = children

        # Finds the best solution in the current generation
        best_distance_current_gen = float('inf')
        best_solution_current_gen = None
        for individual in population:
            distance = calculate_total_distance(individual, distance_matrix)
            if distance < best_distance_current_gen:
                best_solution_current_gen = individual
                best_distance_current_gen = distance
            if distance < best_distance_overall:
                best_solution_overall = individual
                best_distance_overall = distance

        print(f"Generatie {generation}: Beste afstand huidige generatie = {best_distance_current_gen}")

    print(f"Beste algemene oplossing: {best_solution_overall}")
    print(f"Beste algemene afstand: {best_distance_overall}")

    return best_solution_overall, best_distance_overall


######################################################################################################################
"""The memetic algorithm starts here. 
The memetic algorithm is a genetic algorithm that includes a two-opt local search"""
######################################################################################################################

def two_opt(tour, distance_matrix):
    best_distance = calculate_total_distance(tour, distance_matrix)
    for i in range(1, len(tour) - 2):
        for j in range(i + 2, len(tour)):
            if j == len(tour) - 1:  # Special case for the edge that includes the last and first cities
                delta = (distance_matrix[tour[i - 1], tour[j]] +
                         distance_matrix[tour[i], tour[0]] -
                         distance_matrix[tour[i - 1], tour[i]] -
                         distance_matrix[tour[j], tour[0]])
            else:
                delta = (distance_matrix[tour[i - 1], tour[j]] +
                         distance_matrix[tour[i], tour[j + 1]] -
                         distance_matrix[tour[i - 1], tour[i]] -
                         distance_matrix[tour[j], tour[j + 1]])

            # If swapping is better, perform the swap and update the distance
            if delta < 0:
                tour[i:j] = tour[j - 1:i - 1:-1]  # This reverses the tour segment in place
                best_distance += delta
    return tour



def run_memetic_algorithm(filename, population_size, generations, mutation_rate, tournament_size, elite_count=2):
    tsp_data = load_tsp_data(filename)
    distance_matrix = compute_distance_matrix(tsp_data)

    # Initialize population and apply local search (2-opt) to each individual
    population = [two_opt(individual, distance_matrix) for individual in generate_initial_population(population_size, len(tsp_data))]

    best_solution_overall = None
    best_distance_overall = float('inf')

    for generation in range(generations):
        elite = elitism(population, distance_matrix, elite_count)
        winners = tournament_selection(population, distance_matrix, tournament_size, population_size - elite_count)
        parent_pairs = create_parent_pairs(winners)

        children = []
        for parent1, parent2 in parent_pairs:
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)
            children.extend([child1, child2])

        for i in range(len(children)):
            if random.random() < mutation_rate:
                children[i] = swap_mutation(children[i])

        # Use 2-opt on the children
        for i in range(len(children)):
            children[i] = two_opt(children[i], distance_matrix)

        children.extend(elite)

        # Update the population and find the best solution in the current generation
        population = children 
        best_distance_current_gen = float('inf')
        for individual in population:
            distance = calculate_total_distance(individual, distance_matrix)
            if distance < best_distance_current_gen:
                best_distance_current_gen = distance
            if distance < best_distance_overall:
                best_solution_overall = individual
                best_distance_overall = distance

        print(f"Generatie {generation}: Beste afstand huidige generatie = {best_distance_current_gen}")

    print(f"Beste algemene oplossing: {best_solution_overall}")
    print(f"Beste algemene afstand: {best_distance_overall}")

    return best_solution_overall, best_distance_overall


######################################################################################################################
"""Run both the algorithms below."""
######################################################################################################################

# Constants and file names
filename = 'file-tsp.txt'  
filename_2 = 'bays29file-tsp.txt' 
population_size = 50
generations = 1500
mutation_rate = 0.01
tournament_size = 2

def ask_user_for_algorithm_choice():
    print("What algorithm do you want to run?")
    print("1: Evolutionary Algorithm (EA)")
    print("2: Memetic Algorithm (MA)")
    choice = input("Choose (1 or 2): ")
    return choice

def ask_user_for_dataset_choice():
    print("What dataset do you want to run it on?")
    print("1: file-stp.txt")
    print("2: bays29file-tsp.txt")
    choice = input("Choose (1 or 2): ")
    return choice

# Ask the user for the algorithm choice and dataset choice
user_choice = ask_user_for_algorithm_choice()
user_choice_2 = ask_user_for_dataset_choice()

# Run the chosen algorithm and dataset
if user_choice == '1':
    print("Evolutionary Algorithm (EA) will be run...")
    if user_choice_2 == '1':
        print("Evolutionary Algorithm (EA) will be run on file-tsp.txt...")
        best_solution, best_distance = run_genetic_algorithm(
            filename, population_size, generations, mutation_rate, tournament_size
        )
    elif user_choice_2 == '2':
        best_solution, best_distance = run_genetic_algorithm(
            filename_2, population_size, generations, mutation_rate, tournament_size
        )
    else:
        print("Invalid choice. Please choose 1 or 2.")
elif user_choice == '2':
    print("Memetic Algorithm (MA) will be run...")
    if user_choice_2 == '1':
        print("Memetic Algorithm (MA) will be run on bay29file-tsp.txt...")
        best_solution, best_distance = run_memetic_algorithm(
            filename, population_size, generations, mutation_rate, tournament_size
        )
    elif user_choice_2 == '2':
        best_solution, best_distance = run_memetic_algorithm(
            filename_2, population_size, generations, mutation_rate, tournament_size
        )
    else:
        print("Invalid choice. Please choose 1 or 2.")
else:
    print("Invalid choice. Please choose 1 or 2.")
