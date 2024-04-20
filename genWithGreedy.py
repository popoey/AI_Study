import random
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

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
            population.append(individual)
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
            total_cost += distance(city1, city2)
        return total_cost


    def selection(self, fitness_values):
        # Roulette wheel selection
        total_fitness = sum(fitness_values)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_indices = [random.choices(range(self.population_size), weights=selection_probabilities)[0] for _ in range(self.population_size)]
        return [self.population[i] for i in selected_indices]

    def crossover(self, parents):
        # Single-point crossover
        crossover_point = random.randint(1, len(parents[0]) - 1)
        child1 = parents[0][:crossover_point] + [gene for gene in parents[1] if gene not in parents[0][:crossover_point]]
        child2 = parents[1][:crossover_point] + [gene for gene in parents[0] if gene not in parents[1][:crossover_point]]
        return child1, child2

    def mutation(self, individual):
        # Swap mutation
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
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

def greedy_tsp(cities, start_city_idx):
    visited = [False] * len(cities)  # 각 도시의 방문 여부를 저장하는 리스트.
    visited[start_city_idx] = True  # 시작 도시 방문 표시
    total_cost = 0  # 총 비용
    count = 0
    current_city = start_city_idx  # 현재 도시의 인덱스
    path = [start_city_idx]  # 경로 초기화
    
    while len(path) < len(cities):
        min_dist = float('inf')  # 현재까지의 최소 거리
        nearest_city = None  # 가장 가까운 도시의 인덱스
        
        # 현재 도시와 가장 가까운 방문하지 않은 도시를 찾음
        for idx, city in enumerate(cities):
            if not visited[idx]:  # 방문하지 않은 도시 검사
                dist = distance(cities[current_city], city)
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = idx
    
        # 가장 가까운 도시를 방문 리스트에 추가하고 경로에 포함시킴
        visited[nearest_city] = True
        path.append(nearest_city)
        total_cost += min_dist
        current_city = nearest_city
        count += 1
    
    # 시작 도시로 돌아가는 경로 추가
    path.append(start_city_idx)  # 시작 도시로 돌아가는 경로 추가
    total_cost += distance(cities[current_city], cities[start_city_idx])  # 시작 도시로 돌아가는 비용 추가
    count += 1
    
    return path, total_cost, count

# CSV 파일 경로
script_dir = os.path.dirname(__file__)
file_name = "2024_AI_TSP.csv"
file_path = os.path.join(script_dir, file_name)

cities = []

# CSV 파일을 읽기 모드로 열기
with open(file_path, mode='r', newline='') as tsp:
  # CSV 파일을 읽기 위한 reader 객체 생성
  reader = csv.reader(tsp)
  for row in reader:
    cities.append([float(row[0]), float(row[1])])

# Greedy 알고리즘으로 TSP 문제를 해결하는 함수
def greedy_tsp_with_start(cities, start_city_idx):
    return greedy_tsp(cities, start_city_idx)

genetic_algorithm = GeneticAlgorithm(population_size=50, crossover_rate=0.8, mutation_rate=0.01, cities=cities)
for _ in range(100):
    genetic_algorithm.evolve()

best_solution = genetic_algorithm.get_best_solution()
print("Best solution from Genetic Algorithm:", best_solution)
print("Total cost from Genetic Algorithm:", genetic_algorithm.fitness_function(best_solution))

# Greedy 알고리즘으로 Genetic Algorithm의 최종 해 개선
best_solution_cost = float('inf')
final_best_solution = None
for i in range(len(best_solution)):
    greedy_solution, greedy_solution_cost, _ = greedy_tsp_with_start(cities, best_solution[i])
    if greedy_solution_cost < best_solution_cost:
        best_solution_cost = greedy_solution_cost
        final_best_solution = greedy_solution

print("Final Best solution after Greedy Improvement:", final_best_solution)
print("Final Total cost after Greedy Improvement:", best_solution_cost)

# Greedy 알고리즘 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter([city[0] for city in cities], [city[1] for city in cities], c='red')
for i in range(len(final_best_solution) - 1):
    city1 = cities[final_best_solution[i]]
    city2 = cities[final_best_solution[i + 1]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], c='blue')
plt.title('Best Solution (Genetic Algorithm with Greedy Improvement)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
