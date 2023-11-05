import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_filename') 
parser.add_argument('num_iterations') 
parser.add_argument('num_implementations', nargs='?', default=8) 
args = parser.parse_args()
filename = args.input_filename
num_iter = int(args.num_iterations)
num_impl = int(args.num_implementations)

base_command = "nvcc main.cu -o main --std c++11 -DTIMING_KERNEL_TIME -DCHECK_CORRECTNESS"

ordering = {"default-base":2,"default-shared":1,"oldnew-base":4,"oldnew-shared":7,"distances-base":0,"distances-shared":5,"coo-base":3,"coo-shared":6}

configs = {"default-base":["-DBLOCK_SIZE=32"], "default-shared":["-DSHARED_MEMORY","-DBLOCK_SIZE=32","-DSHARED_CAPACITY=256"],
           "oldnew-base":["-DNODES_OLDNEW","-DBLOCK_SIZE=128"], "oldnew-shared":["-DNODES_OLDNEW","-DSHARED_MEMORY","-DBLOCK_SIZE=128","-DSHARED_CAPACITY=1024"],
           "distances-base":["-DNODES_DISTANCES","-DBLOCK_SIZE=512"],"distances-shared":["-DNODES_DISTANCES","-DSHARED_MEMORY","-DBLOCK_SIZE=512","-DSHARED_CAPACITY=2048"],
           "coo-base":["-DCOO","-DBLOCK_SIZE=128"],"coo-shared":["-DCOO","-DSHARED_MEMORY","-DBLOCK_SIZE=64","-DSHARED_CAPACITY=64"]}

temp_file = "temp.txt"



run_command = "./main " + filename + " > " + temp_file
results = {}
cpu_sum_times = 0.0
total_num_iter = 0
for config in configs:
	if(ordering[config] < num_impl):
		compile_command = base_command
		for flag in configs[config]:
			compile_command += " " + flag
		os.system(compile_command)
		
		gpu_times = []
		for _ in range(num_iter):
			os.system(run_command)
			
			with open(temp_file, "r") as f:
				lines = f.readlines()
			gpu_time = float(lines[0].split(" ")[-2])
			cpu_time = float(lines[1].split(" ")[-2])
			cpu_sum_times += cpu_time
			gpu_times.append(gpu_time)
			total_num_iter += 1
			
		min_time = min(gpu_times)
		max_time = max(gpu_times)
		avg_time = sum(gpu_times)/num_iter
		results[config] = {"minimum time":min_time, "maximum time":max_time, "average time":avg_time, "compile command":compile_command}
	
os.system("rm " + temp_file)
cpu_avg_time = cpu_sum_times/total_num_iter
for config in configs:
	if(ordering[config] < num_impl):
		avg_time = results[config]["average time"]
		avg_speedup = cpu_avg_time/avg_time
		results[config]["average speedup"] = avg_speedup



for config in results:
	print(config + ":")
	result = results[config]
	for label in result:
		value = result[label]	
		unit = ""
		if "time" in label:
			unit = " ms"
		elif "speedup" in label:
			unit = "x"
		print(label + ": " + str(value) + unit)
	print()
