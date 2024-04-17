import numpy as np
import csv
import os
import random

def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist

def greedy_tsp(cities):
    visited = [0] * len(cities)  # 각 도시의 방문 여부를 저장하는 리스트. 1이 방문, 0이 미방문.
    path = [0]  # 현재 시작 도시는 0번 인덱스
    visited[0] = 1  # 시작 도시 방문 표시
    total_cost = 0  # 총 비용
    
    current_city = 0  # 현재 도시의 인덱스
    
    while len(path) < len(cities):
        min_dist = float('inf')  # 현재까지의 최소 거리
        nearest_city = None  # 가장 가까운 도시의 인덱스
        
        # 현재 도시와 가장 가까운 방문하지 않은 도시를 찾음
        for i, city in enumerate(cities):
            if visited[i]==0:#0이면 방문 안 된 곳
                dist = distance(cities[current_city], city)
                if dist < min_dist:
                    min_dist = dist
                    nearest_city = i
        
        # 가장 가까운 도시를 방문 리스트에 추가하고 경로에 포함시킴
        visited[nearest_city] = 1
        path.append(nearest_city)
        total_cost += min_dist
        current_city = nearest_city
    
    # 시작 도시로 돌아가는 경로 추가
    # path.append(0)  # 시작 도시로 돌아가는 경로 추가 나중에 유전 알고리즘을 위해 잠시 주석으로 바꿈
    total_cost += distance(cities[current_city], cities[0])  # 시작 도시로 돌아가는 비용 추가

    return path, total_cost




import numpy as np

def partial_shuffle(g_path, num_elements):
    # g_path 리스트를 복사하여 새로운 리스트 생성
    shuffled_path = g_path.copy()
    
    # g_path 리스트의 길이를 구함
    length = len(g_path)
    
    # num_elements 수만큼 랜덤하게 인덱스를 선택
    indices = np.random.choice(length, num_elements, replace=False)
    
    # 선택된 인덱스에 해당하는 g_path의 값을 셔플
    selected_elements = [g_path[i] for i in indices]
    np.random.shuffle(selected_elements)
    
    # 셔플된 값들을 g_path의 해당 인덱스에 대입
    for i, idx in enumerate(indices):
        shuffled_path[idx] = selected_elements[i]
    
    return shuffled_path

def generate_population(g_path, size):
    population = []
    for i in range(size):
        path = g_path
        path = partial_shuffle(path, 4)#일단 임의로 4개의 인덱스를 섞음. 
        population.append(path)
    return population

def calculate_fitness(population, cities):
  fitness = []
  for path in population:
      cost = sum(distance(cities[path[i]], cities[path[i+1]]) for i in range(len(path)-1))

      cost += distance(cities[path[-1]], cities[path[0]])  # 시작 도시로 돌아가는 비용 추가
      fitness.append(1 / cost)  # 비용이 적을수록 높은 적합도를 부여
  return fitness

def selection(population, fitness, num_parents):
  selected_indices = np.argsort(fitness)[-num_parents:]
  return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
  crossover_point = np.random.randint(1, len(parent1))
  child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
  child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
  return child1, child2

def mutation(path, mutation_rate):
  if np.random.rand() < mutation_rate:
      idx1, idx2 = np.random.choice(len(path), 2, replace=False)
      path[idx1], path[idx2] = path[idx2], path[idx1]
  return path

def genetic_algorithm(cities, population_size, num_generations, num_parents, mutation_rate):
  population = generate_population(g_path, population_size)#greedy한 것을 기반으로 섞어서 초기 경로 만듬.
  for _ in range(num_generations): #세대 만큼 반복해서 유전 알고리즘 진행.
      fitness = calculate_fitness(population, cities)
      parents = selection(population, fitness, num_parents)
      new_population = parents.copy()
      while len(new_population) < population_size:
          parent_indices = np.random.choice(len(parents), 2, replace=False)
          parent1, parent2 = [parents[idx] for idx in parent_indices]

          child1, child2 = crossover(parent1, parent2)
          child1 = mutation(child1, mutation_rate)
          child2 = mutation(child2, mutation_rate)
          new_population.extend([child1, child2])
      population = new_population[:population_size]  # 넘친 경우 중복 제거
  best_path = max(population, key=lambda x: calculate_fitness([x], cities)[0])
  best_fitness = calculate_fitness([best_path], cities)[0]
  return best_path, 1 / best_fitness






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

# Greedy 알고리즘을 사용하여 TSP 문제를 해결
g_path, total_cost = greedy_tsp(cities)


# 유전 알고리즘을 사용하여 TSP 문제를 해결
population_size = 50
num_generations = 50
num_parents = 20
mutation_rate = 0.1
best_path, total_cost = genetic_algorithm(cities, population_size, num_generations, num_parents, mutation_rate)



print("유전 알고리즘을 사용한 최적 경로:", best_path)
print("총 비용:", total_cost)
