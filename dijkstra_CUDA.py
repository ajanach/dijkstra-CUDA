import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import networkx as nx
import random
import timeit

# CUDA kernel for Dijkstra's algorithm
kernel = """
    __global__ void dijkstra(float *d, int *p, int *visited, int n, int start)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < n)
        {
            if(visited[i] == 0)
            {
                int j;
                for(j = 0; j < n; j++)
                {
                    if(visited[j] == 1)
                    {
                        int weight = p[i*n + j];
                        if(weight != 0 && d[j] + weight < d[i])
                        {
                            d[i] = d[j] + weight;
                            visited[i] = 1;
                        }
                        else
                        {
                            visited[i] = 1;
                        }
                    }
                }
            }
        }
    }
"""

def parallel_dijkstra_gpu(graph, start):
    n = graph.shape[0]
    # Convert weight matrix to 1D array
    graph = graph.flatten()
    d = np.full(n, np.inf)
    d[start] = 0
    visited = np.zeros(n)
    visited[start] = 1

    # Allocate memory on GPU
    d_gpu = gpuarray.to_gpu(d.astype(np.float32))
    p_gpu = gpuarray.to_gpu(graph.astype(np.int32))
    visited_gpu = gpuarray.to_gpu(visited.astype(np.int32))

    
    # Compile CUDA kernel
    mod = SourceModule(kernel)
    func = mod.get_function("dijkstra")
    
    # Execute kernel
    block_size = (256, 1, 1)
    grid_size = (int(np.ceil(n / block_size[0])), 1)
    for i in range(n):
        func(d_gpu, p_gpu, visited_gpu, np.int32(n), np.int32(start), block=block_size, grid=grid_size)

    # Retrieve results from GPU
    d = d_gpu.get()

    return d

def generate_complex_graph(n, min_weight=1, max_weight=10):
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i != j:
                weight = random.randint(min_weight, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

# algorithm testing:
def test_parallel_dijkstra_gpu(inputNodeNumber):
    graph = generate_complex_graph(inputNodeNumber)
    # print("Number of nodes: ", graph.shape[0])
    # print("Number of edges: ", np.count_nonzero(graph)//2)
    start = 0
    distance = parallel_dijkstra_gpu(graph, start)
    return [graph.shape[0], np.count_nonzero(graph)//2]

# time of execution:
def execution_time(inputNodeNumber):
    time = timeit.timeit(lambda: test_parallel_dijkstra_gpu(inputNodeNumber), number=1)
    return time

# nodeNumber = int(input("Number of nodes: "))
# edgeNumber = test_parallel_dijkstra_gpu(nodeNumber)[1]
# executionTime = execution_time(nodeNumber)
# print(f"Parallel execution of algorithm have {nodeNumber} number of nodes and {edgeNumber} number of edges. \n" \
#       f"Parallel execution of the algorithm using CUDA: {executionTime:.3f}sec. \n")