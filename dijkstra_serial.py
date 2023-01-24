import numpy as np
import networkx as nx
import random
import timeit

def dijkstra(graph, start):
    n = graph.shape[0]
    distances = np.full(n, np.inf)
    distances[start] = 0
    visited = np.zeros(n)
    visited[start] = 1
    for i in range(n):
        if visited[i]:
            for j in range(n):
                if graph[i][j] and distances[j] > distances[i] + graph[i][j]:
                    distances[j] = distances[i] + graph[i][j]
                    visited[j] = 1
    return distances

def generate_complex_graph(n, min_weight=1, max_weight=10):
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i != j:
                weight = random.randint(min_weight, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

def test_serial_dijkstra_gpu(inputNodeNumber):
    graph = generate_complex_graph(inputNodeNumber)
    # print("Number of nodes: ", graph.shape[0])
    # print("Number of edges: ", np.count_nonzero(graph)//2)
    start = 0
    distance = dijkstra(graph, start)
    return [graph.shape[0], np.count_nonzero(graph)//2]

def execution_time(inputNodeNumber):
    time = timeit.timeit(lambda: test_serial_dijkstra_gpu(inputNodeNumber), number=1)
    return time

# nodeNumber = int(input("Number of nodes: "))
# edgeNumber = test_serial_dijkstra_gpu(nodeNumber)[1]
# executionTime = execution_time(nodeNumber)
# print(f"Parallel execution of algorithm have {nodeNumber} number of nodes and {edgeNumber} number of edges. \n" \
#       f"Parallel execution of the algorithm using CUDA: {executionTime:.3f}sec. \n")