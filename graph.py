import dijkstra_CUDA, dijkstra_serial
import matplotlib.pyplot as plt
import tkinter as tk

# Enter node number: 
nodeNumberSerial = int(input("Enter the number of nodes for serial execution of the algorithm: "))
nodeNumberParalel = int(input("Enter the number of nodes for parallel (CUDA) execution of the algorithm: "))

# Execution time of both algorithms:
serialExecutionTime = dijkstra_serial.execution_time(nodeNumberSerial)
cudaExecutionTime = dijkstra_CUDA.execution_time(nodeNumberParalel)

root = tk.Tk()
root.title("Number of edges and execution time")

# Test if algorithm have same number of nodes:
if dijkstra_serial.test_serial_dijkstra_gpu(nodeNumberSerial)[0] == dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberParalel)[0] and \
    dijkstra_serial.test_serial_dijkstra_gpu(nodeNumberSerial)[1] == dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberParalel)[1]:
    nodeCount = dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberSerial)[0]
    edgeCount = dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberParalel)[1]
    output_label = tk.Label(root, height=12, font=('Times New Roman', 14), text = \
        f"The algorithms have the same number of nodes ({nodeCount}) and edges ({edgeCount}). \n" \
        f"Serial execution of algorithm: {serialExecutionTime:.3f}sec. \n" \
        f"Parallel execution of the algorithm using CUDA: {cudaExecutionTime:.3f}sec. \n")
    output_label.pack()

else:
    nodeCountParallel = dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberParalel)[0]
    edgeCountParallel = dijkstra_CUDA.test_parallel_dijkstra_gpu(nodeNumberParalel)[1]
    nodeCountSerial = dijkstra_serial.test_serial_dijkstra_gpu(nodeNumberSerial)[0]
    edgeCountSerial = dijkstra_serial.test_serial_dijkstra_gpu(nodeNumberSerial)[1]
    output_label = tk.Label(root, height=12, font=('Times New Roman', 14), text = \
        f"The algorithms do not have the same number of nodes. \n" \
        f"Serial execution of algorithm have {nodeCountSerial} number of nodes and {edgeCountSerial} number of edges. \n" \
        f"Parallel execution of algorithm have {nodeCountParallel} number of nodes and {edgeCountParallel} number of edges. \n" \
        f"Serial execution of algorithm: {serialExecutionTime:.3f}sec . \n" \
        f"Parallel executon of algorithm: {cudaExecutionTime:.3f}sec. \n")
    output_label.pack()

# X-coordinates of left sides of bars 
left = [1, 2]

# Heights of bars
height = [cudaExecutionTime, serialExecutionTime]

# Labels for bars
tick_label = ['Parallel - CUDA', 'Serial - 1xCPU']

# Plotting a bar chart
plt.bar(left, height, tick_label=tick_label,
        width=0.5, color=['green', 'red'])

# Naming the x-axiss
plt.xlabel('Implementation of algorithm')

# Naming the y-axis
plt.ylabel('Time (sec)')

# Plot title and window title
plt.title('Execution time')
plt.get_current_fig_manager().set_window_title("Execution time")

# Function to show the plot
plt.show()

# Textbox: 
root.mainloop()