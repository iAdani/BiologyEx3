import numpy as np
import random

INPUT_SIZE = 16
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1

MAX_GENERATIONS = 100
POPULATION_SIZE = 120
MUTATION_RATE = 0.1
MUTATION_RATIO = 0.2 
TOP_PERCENTAGE = 0.2

# returns a list of tuples (string,label).
def read_data(filename):
    with open(filename, 'r') as file:
        data = [(line.split()[0], int(line.split()[1])) for line in file]
    return data


# each tuple now contains an encoded string and its corresponding label. (string -> int)
def preprocess_data(data):
    processed_data = [(np.array([int(bit) for bit in string]), label) for string, label in data]
    return processed_data


class NeuralNetwork:
    def __init__(self, weights1=None, weights2=None):
        if weights1 is None and weights2 is None:
            self.weights1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) - 0.5  # Shape: (input_size, hidden_size)
            self.weights2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) - 0.5  # Shape: (hidden_size, output_size)
        else:
            self.weights1 = weights1
            self.weights2 = weights2
        
    def forward(self, x):
        hidden = np.dot(x, self.weights1)  # Shape: (batch_size, hidden_size)
        hidden_activation = self.sigmoid(hidden)  # Shape: (batch_size, hidden_size)
        output = np.dot(hidden_activation, self.weights2)  # Shape: (batch_size, output_size)
        output_activation = self.sigmoid(output)  # Shape: (batch_size, output_size)
        return output_activation

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    def copy(self):
        copy = NeuralNetwork()
        copy.weights1 = self.weights1.copy()
        copy.weights2 = self.weights2.copy()
        return copy


# Evaluates the performance of an individual on a given dataset.
def evaluate_fitness(individual, data):
    correct_predictions = 0
    total_samples = len(data)

    for input_data, label in data:
        predicted_label = 1 if individual.forward(input_data) >= 0.5 else 0
        if label == predicted_label:
            correct_predictions += 1

    return correct_predictions / total_samples


# The algorithm aims to find the best-performing individual
# on a given dataset by iteratively improving the population over multiple generations.
def genetic_algorithm(data):
    population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
    best_fitness_scores = []
    average_fitness_scores = []
    
    for generation in range(MAX_GENERATIONS):
        print(f"Generation Number {generation + 1}")

        # Evaluate fitness
        fitness_scores = [evaluate_fitness(individual, data) for individual in population]
        best_fitness = max(fitness_scores)
        average_fitness = sum(fitness_scores) / len(fitness_scores)

        best_fitness_scores.append(best_fitness)
        average_fitness_scores.append(average_fitness)

        # Check for convergence
        convergence_threshold = 1e-6  # Set a threshold for convergence
        convergence_generations = 10  # Number of generations to check for convergence

        if generation >= convergence_generations:
            last_generations = best_fitness_scores[-convergence_generations:]
            if max(last_generations) - min(last_generations) < convergence_threshold:
                print(f"Converged too soon, please try again.")
                break


        print(f"Best Neural Score:\t{best_fitness * 100 :.2f}%\nNeurals Average Score:\t{average_fitness * 100 :.2f}%")

        # Create new generation through crossover and mutation
        offspring = build_population(population, fitness_scores)
        population = offspring

    best_individual = max(population, key=lambda x: evaluate_fitness(x, data))
    # PercentPerGeneration(best_fitness_scores, worst_fitness_scores, average_fitness_scores)
    return best_individual


def build_population(population, fitness_scores):
    new_population_list = []

    # Sort the population based on fitness scores in descending order
    sorted_population = [individual for _, individual in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]

    # Select the best individual and add it to the new population
    best_individual = sorted_population[0].copy()
    new_population_list.append(best_individual)

    # Add mutated versions of the best individual to the new population
    for _ in range(int(POPULATION_SIZE / 2)):
        mutated_individual = mutate(best_individual.copy())
        new_population_list.append(mutated_individual)

    # Add top individuals to the new population
    top_population = sorted_population[:int(len(sorted_population) * TOP_PERCENTAGE)]
    new_population_list.extend(top_population)

    # Fill the remaining slots in the new population through crossover and mutation
    population.extend(top_population)
    while len(new_population_list) <= POPULATION_SIZE:
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = crossover(parent1, parent2)
        new_population_list.append(mutate(child))

    return new_population_list


def crossover(parent1, parent2):
    child = NeuralNetwork()

    # Perform sum crossover for the weights of the first hidden layer
    child.weights1 = (parent1.weights1 + parent2.weights1) * 0.5

    # Perform sum crossover for the weights of the second hidden layer
    child.weights2 = (parent1.weights2 + parent2.weights2) * 0.5

    # Apply crossover point to both sets of weights
    crossover_point = random.randint(0, parent1.weights1.shape[0])
    child.weights1[crossover_point:] = parent2.weights1[crossover_point:]
    child.weights2[crossover_point:] = parent2.weights2[crossover_point:]

    return child


def mutate(individual):
    for weight_array in [individual.weights1, individual.weights2]:
        mask = np.random.random(size=weight_array.shape) < MUTATION_RATE
        random_values = np.random.uniform(-MUTATION_RATIO, MUTATION_RATIO, size=weight_array.shape)
        weight_array += mask * random_values
    return individual


def weights_to_file(best_individual):
    with open("wnet0.txt", "w") as file:
        # Write the size of the layers
        file.write(f"{INPUT_SIZE} {HIDDEN_SIZE}\n")
        
        # Write weights1
        for row in best_individual.weights1:
            weights_row = " ".join(str(weight) for weight in row)
            file.write(f"{weights_row}\n")
        
        # Write weights2
        for row in best_individual.weights2:
            weights_row = " ".join(str(weight) for weight in row)
            file.write(f"{weights_row}\n")


    
if __name__ == '__main__':
    train_file = "trainset0.txt"
    test_file = "testset0.txt"
    train_array = read_data(train_file)
    train_data = preprocess_data(train_array)
    test_array = read_data(test_file)
    test_data = preprocess_data(test_array)

    best_individual = genetic_algorithm(train_data)

    correct_predictions = 0
    for input_data, label in test_data:
        predicted_label = 1 if best_individual.forward(input_data) >= 0.5 else 0
        if label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy:\t\t\t{accuracy * 100 :.2f}%")
    weights_to_file(best_individual)
