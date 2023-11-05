#ifndef TIMING
#ifdef TIMING_TOTAL_TIME
#define TIMING
#endif
#ifdef TIMING_LOOP_TIME
#define TIMING
#endif
#ifdef TIMING_KERNEL_TIME
#define TIMING
#endif
#endif

#ifdef TIMING
#include <iostream>
#endif
#ifdef TIMING_KERNEL_TIME
#include <vector>
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifdef SHARED_MEMORY
#ifndef SHARED_CAPACITY
#define SHARED_CAPACITY 2048
#endif
#ifndef DEFAULT
#define HALF_SHARED_CAPACITY SHARED_CAPACITY/2
#define CAPACITY_PER_THREAD SHARED_CAPACITY/BLOCK_SIZE
#endif
#endif



#include <stdio.h>
#ifndef SHARED_MEMORY
__constant__ int devNumEdges;
__global__ void coo_gpu_queuing_kernel(const int* __restrict__ firstNodeEdges, const int* __restrict__ secondNodeEdges, const int* __restrict__ oldNodeVisited, int* __restrict__ newNodeVisited, bool* __restrict__ done) {
	//calculate thread index
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for each edge, on a different thread
	if(idx < devNumEdges) {
		const int node1 = firstNodeEdges[idx];
		
		//if the first node of the edge has been visited, mark the second node of the edge as visited and set done to 0
		if(oldNodeVisited[node1]) {
			newNodeVisited[node1] = 1;
			const int node2 = secondNodeEdges[idx];
			if(!oldNodeVisited[node2]) {
				(*done) = false;
				newNodeVisited[node2] = 1;
			}
		}
	}
}
//SHARED_CAPACITY must be an integer multiple of BLOCK_SIZE
#else
__constant__ int devNumEdges;
__constant__ int devNumNodes;
__global__ void coo_gpu_block_queuing_kernel(const int* __restrict__ firstNodeEdges, const int* __restrict__ secondNodeEdges, const int* __restrict__ oldNodeVisited, int* __restrict__ newNodeVisited, bool* __restrict__ done) {
	//calculate thread index
	const int blockIndex = blockIdx.x * blockDim.x;
	const int idx = blockIndex + threadIdx.x;
	const int numNodes = devNumNodes;
	
	//allocate and initialize shared memory
	__shared__ int blockOldNodeVisited[SHARED_CAPACITY];
	__shared__ int blockNewNodeVisited[SHARED_CAPACITY];
	const int beginIndex = min(blockIndex+blockDim.x/2, numNodes-1);
	const int beginBlockNodeVisited = firstNodeEdges[beginIndex];
	const int endBlockNodeVisited = beginBlockNodeVisited+SHARED_CAPACITY;
	int* __restrict__ shiftedBlockOldNodeVisited = blockOldNodeVisited-beginBlockNodeVisited;
	int* __restrict__ shiftedBlockNewNodeVisited = blockNewNodeVisited-beginBlockNodeVisited;
	const int startIndex = threadIdx.x + beginBlockNodeVisited;
	#pragma unroll
	for(int iter=0; iter<CAPACITY_PER_THREAD; iter++) {
		const int index = startIndex + iter*BLOCK_SIZE;
		
		if(index < numNodes) {
			const int value = oldNodeVisited[index];
			shiftedBlockOldNodeVisited[index] = value;
			shiftedBlockNewNodeVisited[index] = value;
		}
	}
	
	//for each edge, on a different thread
	if(idx < devNumEdges) {
		const int node1 = firstNodeEdges[idx];
		const int node2 = secondNodeEdges[idx];
		
		if(node1 >= beginBlockNodeVisited && node1 < endBlockNodeVisited) {
			if(shiftedBlockOldNodeVisited[node1]) {
				shiftedBlockNewNodeVisited[node1] = 1;
				if(node2 >= beginBlockNodeVisited && node2 < endBlockNodeVisited) {
					if(!shiftedBlockOldNodeVisited[node2]) {
						(*done) = false;
						shiftedBlockNewNodeVisited[node2] = 1;
					}
				} else if(!oldNodeVisited[node2]) {
					(*done) = false;
					newNodeVisited[node2] = 1;
				}
			}
		} else if(oldNodeVisited[node1]) {
			newNodeVisited[node1] = 1;
			if(node2 >= beginBlockNodeVisited && node2 < endBlockNodeVisited) {
				if(!shiftedBlockOldNodeVisited[node2]) {
					(*done) = false;
					shiftedBlockNewNodeVisited[node2] = 1;
				}
			} else if(!oldNodeVisited[node2]) {
				(*done) = false;
				newNodeVisited[node2] = 1;
			}
		}
	}
	
	//synchronize and write from shared memory to global memory
	__syncthreads();
	#pragma unroll
	for(int iter=0; iter<CAPACITY_PER_THREAD; iter++) {
		const int index = startIndex + iter*BLOCK_SIZE;
		
		if(index < numNodes) {
			if(shiftedBlockNewNodeVisited[index]) {
				newNodeVisited[index] = 1;
			}
		}
	}
}
#endif

float coo_gpu_traversal(int *firstNodeEdges, int *secondNodeEdges, int *nodeVisited, int numNodes, int numEdges) {	
	//allocate space for additional data structures
	int *newNodeVisited = (int*) malloc(sizeof(int)*numNodes);
	
	//traversal starts at node 0, initialize data structures
	int *oldNodeVisited = nodeVisited;
	oldNodeVisited[0] = 1;
	newNodeVisited[0] = 1;
	for(int i=1; i<numNodes; i++) {
		oldNodeVisited[i] = 0;
		newNodeVisited[i] = 0;
	}
	bool done = false;
	
	//declare variables needed to calculate time
	#ifdef TIMING
	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	#endif
	#ifdef TIMING_TOTAL_TIME
	cudaEventRecord(start);
	#endif
	#ifdef TIMING_KERNEL_TIME
	std::vector<float> kernel_times{};
	#endif
	
	//calculate number of blocks
	unsigned int numBlocks = numEdges / BLOCK_SIZE;
	if(numEdges % BLOCK_SIZE != 0) {
		numBlocks++;
	}
	
	cudaMemcpyToSymbol(devNumEdges, &numEdges, sizeof(int));
	#ifdef SHARED_MEMORY
	cudaMemcpyToSymbol(devNumNodes, &numNodes, sizeof(int));
	#endif
	
	//handle cuda memory
	int *devFirstNodeEdges;
	int *devSecondNodeEdges;
	int *devOldNodeVisited;
	int *devNewNodeVisited;
	bool *devDone;
	cudaMalloc(&devFirstNodeEdges, sizeof(int)*numEdges);
	cudaMalloc(&devSecondNodeEdges, sizeof(int)*numEdges);
	cudaMalloc(&devOldNodeVisited, sizeof(int)*numNodes);
	cudaMalloc(&devNewNodeVisited, sizeof(int)*numNodes);
	cudaMalloc(&devDone, sizeof(bool));
	cudaMemcpy(devFirstNodeEdges, firstNodeEdges, sizeof(int)*numEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(devSecondNodeEdges, secondNodeEdges, sizeof(int)*numEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(devOldNodeVisited, oldNodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devNewNodeVisited, newNodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devDone, &done, sizeof(bool), cudaMemcpyHostToDevice);
	
	//handle timing
	#ifdef TIMING_LOOP_TIME
	cudaEventRecord(start);
	#endif
	
	//call the cuda kernel until the traversal is finished
	int *temp;
	while(!done) {
		//transfer needed data to the device
		done = true;
		cudaMemcpy(devDone, &done, sizeof(bool), cudaMemcpyHostToDevice);
		
		//handle timing
		#ifdef TIMING_KERNEL_TIME
		cudaEventRecord(start);
		#endif
		
		//run the kernel
		#ifdef SHARED_MEMORY
		coo_gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devFirstNodeEdges, devSecondNodeEdges, devOldNodeVisited, devNewNodeVisited, devDone);
		#else
		coo_gpu_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devFirstNodeEdges, devSecondNodeEdges, devOldNodeVisited, devNewNodeVisited, devDone);
		#endif
		
		//handle timing
		#ifdef TIMING_KERNEL_TIME
		cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
		float kernel_time = 0;
   		cudaEventElapsedTime(&kernel_time, start, stop);
		kernel_times.emplace_back(kernel_time);
		time += kernel_time;
		#endif
		
		//transfer needed data from the device and prepare data structures for the next execution
		cudaMemcpy(&done, devDone, sizeof(bool), cudaMemcpyDeviceToHost);
		temp = devOldNodeVisited;
		devOldNodeVisited = devNewNodeVisited;
		devNewNodeVisited = temp;
	}
	
	//handle timing
	#ifdef TIMING_LOOP_TIME
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	#endif
	
	//get the result from the device
	cudaMemcpy(nodeVisited, devNewNodeVisited, sizeof(int)*numNodes, cudaMemcpyDeviceToHost);
	
	//calculate and print timing data
	#ifdef TIMING_TOTAL_TIME
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    #endif
    #ifdef TIMING
    #ifndef TIMING_KERNEL_TIME
    cudaEventElapsedTime(&time, start, stop);
    #endif
    #endif
    #ifdef TIMING_TOTAL_TIME
    std::cout << "GPU time (with data exchange):  " << time << " ms" << std::endl;
    #endif
    #ifdef TIMING_LOOP_TIME
    std::cout << "GPU time (no data exchange):    " << time << " ms" << std::endl;
    #endif
    #ifdef TIMING_KERNEL_TIME
    #ifdef TIMING_FULL_LOGGING
    std::cout << "Time for each kernel invocation: " << std::endl;
    for(int i=0; i<kernel_times.size(); i++) {
    	std::cout << i+1 << "-th invocation:                " << kernel_times[i] << " ms\n";
    }
    #endif
    std::cout << "GPU time (sum of kernel times): " << time << " ms" << std::endl;
    #endif
    
    #ifdef TIMING
    return time;
    #else
    return NAN;
	#endif
}
