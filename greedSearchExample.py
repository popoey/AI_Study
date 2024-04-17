import numpy as np
import csv
import os

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
    path.append(0)  # 시작 도시로 돌아가는 경로 추가
    total_cost += distance(cities[current_city], cities[0])  # 시작 도시로 돌아가는 비용 추가

    return path, total_cost

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
path, total_cost = greedy_tsp(cities)
print("Greedy 알고리즘을 사용한 최적 경로:", path)
print("총 비용:", total_cost)
