import random
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import heapq
class AStar:
    def __init__(self, start, goal, heuristic, distance):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.heuristic = heuristic
        self.distance = distance
        self.came_from = {}

    def search(self):
        open_set = [(self.heuristic(self.start, self.goal), self.start)]
        closed_set = set()

        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == self.goal:
                path = self.reconstruct_path(current)
                return path

            closed_set.add(current)

            for neighbor in self.neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < g_score.get(neighbor, float('inf')) or neighbor not in [i[1] for i in open_set]:
                    self.came_from[neighbor] = current  # Update came_from
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def neighbors(self, node):
        # Define how to get neighbors of a node
        neighbors = []
        for i in range(len(node)):
            for j in range(i + 1, len(node)):
                neighbor = list(node)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(tuple(neighbor))
        return neighbors

    def reconstruct_path(self, current):
        # Reconstruct the path from the current node to the start node
        path = [current]
        while current != self.start:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]



# Example usage
def euclidean_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

def heuristic(city1, city2):
    return euclidean_distance(city1, city2)

# CSV 파일을 읽어와서 cities 변수 초기화
script_dir = os.path.dirname(__file__)
file_name = "2024_AI_TSP.csv"
file_path = os.path.join(script_dir, file_name)

cities = []

with open(file_path, mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append([float(row[0]), float(row[1])])

start_city = cities[0]
goal_city = cities[-1]

astar = AStar(start_city, goal_city, heuristic, euclidean_distance)
path = astar.search()

print("A* Path:", path)

class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, cities):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cities = cities
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize population randomly
        population = []
        for _ in range(self.population_size):
            individual = list(range(len(self.cities)))
            random.shuffle(individual)
            population.append(tuple(individual))
        return population

    def evaluate_fitness(self):
        # Evaluate fitness of each individual in the population
        fitness_values = []
        for individual in self.population:
            fitness_values.append(self.fitness_function(individual))
        return fitness_values

    def fitness_function(self, solution):
        total_cost = 0
        for i in range(len(solution) - 1):
            city1 = self.cities[solution[i]]
            city2 = self.cities[solution[i + 1]]
            total_cost += self.distance(city1, city2)
        return total_cost

    def distance(self, city1, city2):
        return np.linalg.norm(np.array(city1) - np.array(city2))

    def selection(self, fitness_values):
        # Roulette wheel selection
        total_fitness = sum(fitness_values)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_indices = [random.choices(range(self.population_size), weights=selection_probabilities)[0] for _ in range(self.population_size)]
        return [self.population[i] for i in selected_indices]

    def crossover(self, parents):
        # Single-point crossover
        crossover_point = random.randint(1, len(parents[0]) - 1)
        child1 = parents[0][:crossover_point] + tuple(gene for gene in parents[1] if gene not in parents[0][:crossover_point])
        child2 = parents[1][:crossover_point] + tuple(gene for gene in parents[0] if gene not in parents[1][:crossover_point])
        return child1, child2

    def mutation(self, individual):
        # Swap mutation
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual_list = list(individual)
            individual_list[idx1], individual_list[idx2] = individual_list[idx2], individual_list[idx1]
            return tuple(individual_list)
        return individual

    def evolve(self):
        fitness_values = self.evaluate_fitness()
        selected_population = self.selection(fitness_values)
        next_population = []
        while len(next_population) < self.population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover([parent1, parent2])
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                next_population.extend([child1, child2])
            else:
                next_population.extend([parent1, parent2])
        self.population = next_population

    def get_best_solution(self):
        fitness_values = self.evaluate_fitness()
        best_index = fitness_values.index(min(fitness_values))
        return self.population[best_index]

# Example usage
genetic_algorithm = GeneticAlgorithm(population_size=50, crossover_rate=0.8, mutation_rate=0.01, cities=cities)
for _ in range(100):
    genetic_algorithm.evolve()

best_solution = genetic_algorithm.get_best_solution()
print("Best solution:", best_solution)
print("Total cost:", genetic_algorithm.fitness_function(best_solution))

# Plotting the best solution
plt.figure(figsize=(8, 8))
plt.scatter([city[0] for city in cities], [city[1] for city in cities], c='red')
for i in range(len(best_solution) - 1):
    city1 = cities[best_solution[i]]
    city2 = cities[best_solution[i + 1]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], c='blue')
plt.title('Best Solution (Genetic Algorithm)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
