import os
import csv
import random
import math
import numpy as np
from bitarray import bitarray #pip install bitarray 이렇게 터미널에 입력해서 모듈 설치하고 해야 함, 이거 말고도 외부 모듈은 pip install 모듈명으로 터미널에 입력해줘야 함.
import time#시간 측정용
import matplotlib.pyplot as plt



#전체적인 아이디어는 초기해를 greedy search를 통해 구하고, 그것으로 유전 알고리즘을 하는 방법. 거기에 추가적으로 계속 현재 제일 좋은 해를 가지고 가며 일정 세대수가 지나도 딱히 변화가 없으면 다시 현재 제일 좋은 해로 초기해를 구축함. => 그래서 아마 지역 최적에 빠질 가능성이 커 보임. 그래서 실행에 따라 cost가 줄어드는 게 달라짐.=> 더 좋은 방법이 있으면 언제든 환영 
#이 방법은 선택을 아예 랜덤하게 만듬. 다른 선택 방법보다 아예 pruning을 해서 확인할 필요 없는 건 넘어가고 세대수를 더 빠르게 진행하는 게 낫지 않을까 싶은 생각 + limit_generation_without_improvement사이클 동안 조금이라도 전역 최적으로 갈 가능성 때문에 그렇게 만듬. 전통적인 GA는 pruning 떼고 선택 방식 바꾸는 게 맞을듯
#코드 최적화나 구현이 아직 덜 됨. 예를 들어 비트 연산, doubly linked list로 해서 속도를 더 빠르게 해준다든가, mutation_rate을 동적으로 서서히 줄인다든가. mutation 개수를 늘려보든가, 초기 num_generation, mutation_rate, crossover_rate, population_size, limit_generation_without_improvement등 여러 설정들을 조정한다든가. 돌아오는 경로까지 포함해서 최적을 구한다든가, 다점 교차를 실험해본다든가, 기타 등등
#실행 속도를 빠르게 하기 위해선 확인용으로 만든 출력문을 다 지우는 게 훨씬 빠름. 대신 바뀌는 걸 못 봐서 재미가 없음
#population을 늘리면 정말 빠르게 지역 최적에 감. 대신 엄청 느림.



start_time = time.time()#시간 측정용

# 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_name = "2024_AI_TSP.csv"
file_path = os.path.join(script_dir, file_name)

# 도시 정보 저장할 리스트
cities = []

# CSV 파일 읽기
with open(file_path, mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append([float(row[0]), float(row[1])])


def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist


# 거리 계산 함수
class greedyAlgorithm:
  def __init__(self,cities):
    self.cities = cities
    
  def greedy_search(self):
    tot_cost = 0  # 총 비용
    current_city = 0  # 현재 도시의 인덱스
    path = [0]  # 경로 초기화
    num_cities = len(cities)
    visited= bitarray(num_cities)#비트마스킹
    visited[0] = 1 #방문 표시, 처음 인덱스는 방문
    
    while len(path) < num_cities:
      min_dist = math.inf
      nearby_city = None#현재 가장 근처에 있는 도시

      for next_city in range(num_cities):
        if not visited[next_city]:
          dist = distance(cities[current_city], cities[next_city])
          if dist < min_dist:
            min_dist = dist
            nearby_city = next_city
      tot_cost += min_dist
      visited[nearby_city] = 1
      path.append(nearby_city)
      current_city = nearby_city

    return path, tot_cost



class geneticAlgorithm:
  def __init__(self, cities, initial_cost, initial_path, population_size, crossover_rate, mutation_rate, num_generation):
    self.cities = cities
    self.best_cost = initial_cost#처음은 최고의 cost가 초기 greedy_seach를 통해 얻은 cost이다.
    self.best_path = initial_path#처음은 최고의 경로가 초기 greedy_seach를 통해 얻은 경로이다.
    self.population_size = population_size
    self.crossover_rate = crossover_rate
    self.mutation_rate=mutation_rate
    self.num_generation = num_generation
    self.limit_generation_without_improvement = 100  # 일정 세대 동안 해가 개선되지 않으면 초기해를 다시 셔플하여 새로운 개체를 생성합니다.
    self.generations_without_improvement = 0  # 해가 개선되지 않은 세대 수를 추적합니다.


  def generate_population(self, individual):#최고의 경로를 바탕으로 그 값을 조금씩 셔플한 것을 초기해들로 삼는다.
    population = []
    for _ in range(self.population_size):
      individual = self.best_path.copy()  # 최고의 경로를 복사하여 사용
      shuffled_path = self.mutation(individual)
      population.append(shuffled_path)
    return population
  
  def mutation(self, individual):#들어온 리스트를 한 번 셔플한다.
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

  def crossover(self, parent1, parent2):
      # 일점교차
      crossover_point = random.randint(1, len(parent1) - 1)
      child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
      return child1

  def fitness_function(self, individual, current_best_solution_cost):
        #경로의 모든 도시간의 거리를 더한다. 
    total_cost = 0
    for i in range(len(individual) - 1):
      city1 = self.cities[individual[i]]
      city2 = self.cities[individual[i + 1]]
      total_cost += distance(city1, city2)
      if total_cost >= current_best_solution_cost:
        return 0
    print(i,"세대 성능이 좋아졌습니다! cost:", current_best_solution_cost)  
    return total_cost

  def evolve(self):
    current_best_solution_path = self.best_path# 현재 가장 최고의 경로
    current_best_solution_cost  = self.best_cost# 현재 가장 최고의 cost
    current_population = self.generate_population(current_best_solution_path)#종합 경로 개체들 리스트
    for i in range(self.num_generation):
      next_population = []#루프 안에서의 경로 개체들 리스트
      while len(next_population) < self.population_size: #population의 수를 보존   
        parent1, parent2 = random.sample(current_population, 2)  # 서로 다른 두 부모 선택
        if random.random() < self.crossover_rate:
          child= self.crossover(parent1, parent2)
          if random.random() < self.mutation_rate:
            child = self.mutation(child)
          next_population.append(child)
        else:
          if parent1 not in next_population:
            next_population.append(parent1)
          if parent2 not in next_population:  
            next_population.append(parent2)

        # 마지막 개체가 population_size를 초과하는 경우 제거
      next_population = next_population[:self.population_size]
      current_population = next_population #이게 루프의 종합 개체들

      # 현재 세대의 개체들 중에서 최적해 찾기
      for idx in range(len(next_population)):
        next_best_cost = self.fitness_function(current_population[idx], current_best_solution_cost)
        if next_best_cost != 0:
          current_best_solution_cost = next_best_cost
          current_best_solution_path = current_population[idx]
          self.generations_without_improvement = 0
          print(i,"세대 성능이 좋아졌습니다! cost:", current_best_solution_cost)

      self.generations_without_improvement += 1

      # 일정 세대 동안 해가 개선되지 않으면 현재 최적해를 다시 섞어서 새로운 개체를 생성
      if self.generations_without_improvement >= self.limit_generation_without_improvement:
        current_population = self.generate_population(current_best_solution_path)
        self.generations_without_improvement = 0
        print(i, "세대 개체를 새로 생성하였습니다!")

      best_solution_cost = current_best_solution_cost
      best_solution_path = current_best_solution_path

    return best_solution_cost, best_solution_path


#여기부터 메인 코드 시작
gs = greedyAlgorithm(cities)
initial_path, initial_cost = gs.greedy_search()

#print("greedy search를 사용한 초기 경로:", initial_path)
print("greedy search를 사용한 초기 비용:", initial_cost)

population_size = 30
crossover_rate =0.7
mutation_rate =0.1
num_generation = 1000

ga = geneticAlgorithm(cities,initial_cost, initial_path, population_size, crossover_rate, mutation_rate, num_generation)
best_solution_cost, best_solution_path = ga.evolve()#현재 여기서 path가 줄어드는 버그가 생김.

#print("GA를 사용한 최종 경로:", best_solution_path)
print("GA를 사용한 최종 비용:", best_solution_cost)


#최종 solution 저장

script_dir = os.path.dirname(__file__)
file_name = "solution.csv"
file_path = os.path.join(script_dir, file_name)

with open(file_path, mode='w', newline='') as solution:
    writer = csv.writer(solution)
    for row in best_solution_path:
        writer.writerow([row])

#여기서부턴 교수님 코드
sol = []

with open(file_path, mode='r', newline='') as solution:
    # CSV 파일을 읽기 위한 reader 객체 생성
    reader = csv.reader(solution)
    for row in reader:
        sol.append(int(row[0]))


#정렬 후 시작
idx = sol.index(0)

front = sol[idx:]
back = sol[0:idx]

sol = front +back

sol.append(int(0))

total_cost = 0
    
for idx in range(len(sol)-1):
    
    #get city positions
    pos_city_1 = [float(cities[sol[idx]][0]), float(cities[sol[idx]][1])]
    pos_city_2 = [float(cities[sol[idx+1]][0]), float(cities[sol[idx+1]][1])]

    #distance calculation
    dist = distance(pos_city_1, pos_city_2) 

    #accumulation
    total_cost +=dist

print('final cost: '+str(total_cost))




#시간 측정용
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")

# 알고리즘 결과 도트로 표현
plt.figure(figsize=(8, 8))
plt.scatter([city[0] for city in cities], [city[1] for city in cities], c='red')
for i in range(len(sol) - 1):
    city1 = cities[sol[i]]
    city2 = cities[sol[i + 1]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], c='blue')
plt.title('Best Solution (Genetic Algorithm)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

