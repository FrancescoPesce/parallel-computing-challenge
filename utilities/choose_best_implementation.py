import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_filename') 
parser.add_argument('num_iterations') 
args = parser.parse_args()
filename = args.input_filename
num_iter = int(args.num_iterations)

#parsing
with open(filename, "r") as f:
	data = f.readline()

words = data.split(" ")
numNodes = int(words[0])
#edges from a node to itself should also be removed, ignored here
numEdges = 2*int(words[1])

#sparsity = numEdges/numNodes**2
edges_node = numEdges/numNodes

#only three possible implementations to reduce degrees of freedom
implementations = ["distances-base", "default-shared", "coo-base"]
block_size = {"default-shared":32, "distances-base":128, "coo-base":128}
shared_memory = {"default-shared":2048}
impl_flags = {"distances-base":"-DNODES_DISTANCES", "default-shared":"-DSHARED_MEMORY", "coo-base":"-DCOO"}

temp_file = "temp.txt"

#using the following data to choose how to set the parameters, with more emphasis on the first 5 since they are from an official NVIDIA dataset
"""
kkt_power 2M nodes, 3.8e-6 sparsity, 7.9 nnz row -> default-shared/distances-base
coPapersCiteseer 4.3k nodes, 1.7e-2 sparsity, 74 nnz row -> distances-base default-shared
audikw_1 9.4k nodes, 8.8e-3 sparsity, 83.28 nnz row -> distances-base default-shared
wikipedia-20070206 3.6M nodes, 6.7e-6 sparsity, 24 nnz row -> distances-base
kron_g500-logn20 1M nodes, 8.1e-5 sparsity, 85 nnz row -> coo-base distances-base

roadNet-CA 2M nodes, 1.4e-6 sparsity, 2.8 nnz row -> default-shared
bcsstk30 29k nodes, 2.4e-3 sparsity, 71.6 nnz row -> distances-base

standard 5k nodes, 8e-4 sparsity, 4 nnz row -> coo-base distances-base
standard2 50k nodes, 8e-5 sparsity, 4 nnz row -> distances-base/default-shared
standard3 100k nodes, 4e-5 sparisty, 4 nnz row -> default-shared/distances-base
standard4 250k nodes, 1.6e-5 sparsity, 4 nnz row -> default-shared/distances-base
standard5 1M nodes, 4e-6 sparsity, 4 nnz row -> distances-base default-shared
standard6 2M nodes, 2e-6 sparsity, 4 nnz row -> distances-base default-shared

a0nsdsil 80k nodes, 6.2e-5 sparisty, 5 nnz row -> coo-base distances-base
cavity15 2.6k nodes, 0.012 sparsity, 31.9 nnz row -> coo-base distances-base
chipcool1 20k nodes, 7.5e-4 sparsity, 15 nnz row -> distances-base coo-base/default-shared
kineticBatchReactor_2 4.4k nodes, 2.5e-3 sparsity, 10.8 nnz row -> coo-base distances-base
lnsp3937 4k nodes, 2e-3 sparsity, 8.1 nnz row -> coo-base/distances-base
M20PI_n1 1k nodes, 2.9e-3 sparsity, 2.95 nnz row -> all similar
poli_large 16k nodes, 2.7e-4 sparsity, 4.24 nnz row -> coo-base distances-base
Wordnet3 83k nodes, 3.5e-5 sparsity, 2.9 nnz row -> coo-base distances-base

cit-HepTh 28k nodes, 9.1e-4 sparsity, 25.4 nnz row -> coo-base distances-base
web-Stanford 280k nodes, 5e-5 sparsity, 14.14 nnz row-> distances-base coo-base
wiki-Talk 2.4M nodes, 1e-6 sparsity, 3.9 nnz row -> coo-base distances-base
"""

#I decided to never choose "coo-base" because the variance is too high to use it in a competition 
if edges_node > 3.85:
	implementation = "distances-base"
else:
	implementation = "default-shared"
	


base_command = "nvcc parallel-computing-challenge/src/main.cu -o main --std c++11 -DTIMING_KERNEL_TIME -DCHECK_CORRECTNESS"
compile_command = base_command + " " + impl_flags[implementation] + " -DBLOCK_SIZE=" + str(block_size[implementation])
if "shared" in implementation:
	compile_command = compile_command + " -DSHARED_CAPACITY=" + str(shared_memory[implementation])

run_command = "./main " + filename + " > " + temp_file

results = {}

os.system(compile_command)
gpu_times = []
cpu_sum_times = 0
for _ in range(num_iter):
	os.system(run_command)
	
	with open(temp_file, "r") as f:
		lines = f.readlines()
	gpu_time = float(lines[0].split(" ")[-2])
	cpu_time = float(lines[1].split(" ")[-2])
	cpu_sum_times += cpu_time
	gpu_times.append(gpu_time)
os.system("rm " + temp_file)
min_time = min(gpu_times)
max_time = max(gpu_times)
avg_time = sum(gpu_times)/num_iter
avg_cpu_time = cpu_sum_times/num_iter
avg_speedup = cpu_sum_times/avg_time
results = {"minimum time":min_time, "maximum time":max_time, "average time":avg_time, "compile command":compile_command, "average speedup":avg_speedup}

for label in results:
	value = results[label]	
	unit = ""
	if "time" in label:
		unit = " ms"
	elif "speedup" in label:
		unit = "x"
	print(label + ": " + str(value) + unit)


