import os
from math import log2

base_command = "nvcc parallel-computing-challenge/src/main.cu -o main --std c++11 -DTIMING_KERNEL_TIME"

flag_configs = {"default-base":[], "default-shared":["-DSHARED_MEMORY"],
           "oldnew-base":["-DNODES_OLDNEW"], "oldnew-shared":["-DNODES_OLDNEW","-DSHARED_MEMORY"],
           "distances-base":["-DNODES_DISTANCES"],"distances-shared":["-DNODES_DISTANCES","-DSHARED_MEMORY"],
           "coo-base":["-DCOO"],"coo-shared":["-DCOO","-DSHARED_MEMORY"]}      
           
block_sizes = [32, 64, 128, 256, 512, 1024]
max_mem_size = 8192 #coo: max is calculated as half of this amount, default: there is no need for it to be a power of 2, but only powers of two will be tested.
       
num_iter = 1

filename = "parallel-computing-challenge/resources/bcsstk30_parsed.mtx"
out_filename = "res.txt"

temp_file = "temp.txt"



configs = {}
for flag_config in flag_configs:
	for block_size in block_sizes:
		if "shared" in flag_config:
			lower_limit = int(log2(block_size))
			upper_limit = int(log2(max_mem_size))
			if "coo" in flag_config:
				upper_limit -= 1
			for mem_size in [2**i for i in range(lower_limit,upper_limit+1)]:
				name = flag_config + "-block" + str(block_size) + "-shared" + str(mem_size)
				flags = flag_configs[flag_config].copy()
				flags.append("-DBLOCK_SIZE=" + str(block_size))
				flags.append("-DSHARED_CAPACITY=" + str(mem_size))
				configs[name] = flags
		else:
			name = flag_config + "-block" + str(block_size)
			flags = flag_configs[flag_config].copy()
			flags.append("-DBLOCK_SIZE=" + str(block_size))
			configs[name] = flags



run_command = "./main " + filename + " > " + temp_file
results = {}
cpu_sum_times = 0.0
total_num_iter = 0
for config in configs:
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
	results[config] = {"minimum time":min_time, "maximum time":max_time, "average time":avg_time}

os.system("rm " + temp_file)
cpu_avg_time = cpu_sum_times/total_num_iter
for config in configs:
	avg_time = results[config]["average time"]
	avg_speedup = cpu_avg_time/avg_time
	results[config]["average speedup"] = avg_speedup



with open(out_filename, "w") as f:
	for config in results:
		f.write(config + ":\n")
		result = results[config]
		for label in result:
			value = result[label]	
			unit = ""
			if "time" in label:
				unit = " ms"
			elif "speedup" in label:
				unit = "x"
			f.write(label + ": " + str(value) + unit + "\n")
		f.write("\n")

