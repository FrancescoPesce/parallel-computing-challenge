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

#ifndef NODES_OLDNEW
#ifndef NODES_DISTANCES
#define DEFAULT
#endif
#endif



#ifdef DEFAULT
#ifndef SHARED_MEMORY
__global__ void gpu_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, int* __restrict__ nodeVisited, const int* __restrict__ currLevelNodes, int* __restrict__ nextLevelNodes, const int numCurrLevelNodes, int* __restrict__ numNextLevelNodes) {
	//calculate thread index
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for each node of the current level, on a different thread
	if(idx < numCurrLevelNodes) {
		const int node = currLevelNodes[idx];
		const int startNbrIdx=nodePtrs[node];
		const int endNbrIdx=nodePtrs[node+1];
		//loop over all neighbors of the node
		for(int nbrIdx=startNbrIdx; nbrIdx<endNbrIdx; ++nbrIdx) {
			const int neighbor = nodeNeighbors[nbrIdx];
			int* __restrict__ nodeVisitedNeighborAddr = nodeVisited+neighbor;
			//if the neighbor has already been enqueued, pass, otherwise unqueue it
			if(!atomicCAS(nodeVisitedNeighborAddr,0,1)) {
				const int queuePosition = atomicAdd(numNextLevelNodes,1);
				nextLevelNodes[queuePosition] = neighbor;
			}
		}
	}
}
#else
__global__ void gpu_block_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, int* __restrict__ nodeVisited, const int* __restrict__ currLevelNodes, int* __restrict__ nextLevelNodes, const int numCurrLevelNodes, int* __restrict__ numNextLevelNodes) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//shared memory allocation and initialization
	extern __shared__ int blockNextLevelNodes[];
	__shared__ int blockNumNextLevelNodes;
	__shared__ int startNextLevelIdx;
	blockNumNextLevelNodes = 0;
	__syncthreads();
	
	//like the version without shared memory, but queueing is done in shared memory
	if(idx < numCurrLevelNodes) {
		const int node = currLevelNodes[idx];
		const int startNbrIdx=nodePtrs[node];
		const int endNbrIdx=nodePtrs[node+1];
		for(int nbrIdx=startNbrIdx; nbrIdx<endNbrIdx; ++nbrIdx) {
			const int neighbor = nodeNeighbors[nbrIdx];
			int* __restrict__ nodeVisitedNeighborAddr = nodeVisited+neighbor;
			if(!atomicCAS(nodeVisitedNeighborAddr,0,1)) {
				const int queuePosition = atomicAdd(&blockNumNextLevelNodes,1);
				if(queuePosition < SHARED_CAPACITY) {
					blockNextLevelNodes[queuePosition] = neighbor;
				} else {
					const int globalQueuePosition = atomicAdd(numNextLevelNodes,1);
					nextLevelNodes[globalQueuePosition] = neighbor;
				}
			}
		}
	}
	
	//find index in global queue to start copying and synch threads
	__syncthreads();
	blockNumNextLevelNodes = min(blockNumNextLevelNodes,SHARED_CAPACITY);
	if(threadIdx.x == 0) {
		startNextLevelIdx = atomicAdd(numNextLevelNodes,blockNumNextLevelNodes);
	}
	__syncthreads();
	
	//copy data into global queue
	int* __restrict__ shiftedNextLevelNodes = nextLevelNodes+startNextLevelIdx;
	const int numThreadIter = blockNumNextLevelNodes / BLOCK_SIZE + 1;
	for(int iter=0; iter<numThreadIter; iter++) {
		const int index = threadIdx.x + iter*BLOCK_SIZE;
		if(index < blockNumNextLevelNodes) {
			shiftedNextLevelNodes[index] = blockNextLevelNodes[index];
		}
	}
}
#endif
#endif

#ifdef NODES_OLDNEW
__constant__ int devNumNodes;
#ifndef SHARED_MEMORY
__global__ void gpu_oldnew_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, const int* __restrict__ oldNodeVisited, int* __restrict__ newNodeVisited, bool* __restrict__ done) {
	//calculate thread index
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//for each node of the current level, on a different thread
	if(idx < devNumNodes) {
		if(!oldNodeVisited[idx]) {
			const int beginNeighborIdx = nodePtrs[idx];
			const int endNeighborIdx = nodePtrs[idx+1];
			for(int nbrIdx=beginNeighborIdx; nbrIdx<endNeighborIdx; ++nbrIdx) {
				const int neighbor = nodeNeighbors[nbrIdx];
				if(oldNodeVisited[neighbor]) {
					newNodeVisited[idx] = 1;
					(*done) = false;
					break;
				}
			}
		} else {
			newNodeVisited[idx] = 1;
		}
	}
}
#else
//assuming SHARED_CAPACITY is even and an integer multiple of BLOCK_SIZE
__global__ void gpu_oldnew_block_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, const int* __restrict__ oldNodeVisited, int* __restrict__ newNodeVisited, bool* __restrict__ done) {
	//calculate thread index
	const int firstBlockIdx = blockIdx.x * blockDim.x;
	const int halfBlockIdx = firstBlockIdx + blockDim.x/2;
	const int idx = firstBlockIdx + threadIdx.x;
	const int numNodes = devNumNodes;
	
	//allocate and initialize shared memory
	__shared__ int blockOldNodeVisited[SHARED_CAPACITY];
	const int beginNodeIdx = max(0, halfBlockIdx-HALF_SHARED_CAPACITY);
	const int endNodeIdx = min(numNodes, halfBlockIdx+HALF_SHARED_CAPACITY);
	int* __restrict__ shiftedBlockOldNodeVisited = blockOldNodeVisited-beginNodeIdx;
	const int baseThreadIdx = beginNodeIdx + threadIdx.x;
	#pragma unroll
	for(int iter=0; iter<CAPACITY_PER_THREAD; iter++) {
		const int index = baseThreadIdx + iter*BLOCK_SIZE;
		shiftedBlockOldNodeVisited[index] = oldNodeVisited[index];
	}
	__syncthreads();
	
	//for each node of the current level, on a different thread, using shared memory when possible
	if(idx < numNodes) {
		if(!shiftedBlockOldNodeVisited[idx]) {
			const int beginNeighborIdx = nodePtrs[idx];
			const int endNeighborIdx = nodePtrs[idx+1];
			for(int nbrIdx=beginNeighborIdx; nbrIdx<endNeighborIdx; ++nbrIdx) {
				const int neighbor = nodeNeighbors[nbrIdx];
				if(neighbor >= beginNodeIdx && neighbor < endNodeIdx) {
					if(shiftedBlockOldNodeVisited[neighbor]) {
						newNodeVisited[idx] = 1;
						(*done) = false;
						break;
					}
				} else if(oldNodeVisited[neighbor]) {
					newNodeVisited[idx] = 1;
					(*done) = false;
					break;
				}
			}
		} else {
			newNodeVisited[idx] = 1;
		}
	}
}
#endif
#endif

#ifdef NODES_DISTANCES
__constant__ int devNumNodes;
#ifndef SHARED_MEMORY
__global__ void gpu_distances_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, int* __restrict__ nodeDistances, bool* __restrict__ done, const int level) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < devNumNodes) {
		if (nodeDistances[idx] == level) {
			const int beginNeighborIdx = nodePtrs[idx];
			const int endNeighborIdx = nodePtrs[idx+1];
			for(int nbrIdx=beginNeighborIdx; nbrIdx<endNeighborIdx; ++nbrIdx) {
				const int neighbor = nodeNeighbors[nbrIdx];
				int* __restrict__ nodeDistancesNeighborAddr = nodeDistances+neighbor;
				if ((*nodeDistancesNeighborAddr) == -1) {
					(*done) = false;
					(*nodeDistancesNeighborAddr) = level + 1;
				}
			}
		}
	}
}
#else
//assuming SHARED_CAPACITY is even and an integer multiple of BLOCK_SIZE
__global__ void gpu_distances_block_queuing_kernel(const int* __restrict__ nodePtrs, const int* __restrict__ nodeNeighbors, int* __restrict__ nodeDistances, bool* __restrict__ done, const int level) {
	//calculate thread index
	const int firstBlockIdx = blockIdx.x * blockDim.x;
	const int halfBlockIdx = firstBlockIdx + blockDim.x/2;
	const int idx = firstBlockIdx + threadIdx.x;
	const int numNodes = devNumNodes;
	
	//allocate and initialize shared memory
	__shared__ int blockNodeDistances[SHARED_CAPACITY];
	const int beginNodeIdx = max(0, halfBlockIdx-HALF_SHARED_CAPACITY);
	const int endNodeIdx = min(numNodes, halfBlockIdx+HALF_SHARED_CAPACITY);
	int* __restrict__ shiftedBlockNodeDistances = blockNodeDistances-beginNodeIdx;
	const int baseThreadIdx = beginNodeIdx + threadIdx.x;
	#pragma unroll
	for(int iter=0; iter<CAPACITY_PER_THREAD; iter++) {
		const int index = baseThreadIdx + iter*BLOCK_SIZE;
		if(index < numNodes) {
			shiftedBlockNodeDistances[index] = nodeDistances[index];
		}
	}
	__syncthreads();
	
	if (idx < numNodes) {
		if (shiftedBlockNodeDistances[idx] == level) {
			const int beginNeighborIdx = nodePtrs[idx];
			const int endNeighborIdx = nodePtrs[idx+1];
			for(int nbrIdx=beginNeighborIdx; nbrIdx<endNeighborIdx; ++nbrIdx) {
				const int node = nodeNeighbors[nbrIdx];
				if(node >= beginNodeIdx && node < endNodeIdx) {
					if(shiftedBlockNodeDistances[node] == -1) {
						(*done) = false;
						nodeDistances[node] = level + 1;
					}
				} else if (nodeDistances[node] == -1) {
					(*done) = false;
					nodeDistances[node] = level + 1;
				}
			}
		}
	}
}
#endif
#endif

float gpu_traversal(int *nodePtrs, int *nodeNeighbors, int *nodeVisited, int numNodes, int numEdges) {	
	//allocate space for additional data structures
	bool done;
	#ifdef DEFAULT
	int *currLevelNodes = (int*) malloc(sizeof(int)*numNodes);
	int *nextLevelNodes = (int*) malloc(sizeof(int)*numNodes);
	#endif
	
	//traversal starts at node 0, initialize data structures
	#ifdef DEFAULT
	currLevelNodes[0] = 0;
	int numCurrLevelNodes = 1;
	int numNextLevelNodes = 0;
	#endif
	#ifndef NODES_DISTANCES
	nodeVisited[0] = 1;
	for(int i=1; i<numNodes; i++) {
		nodeVisited[i] = 0;
	}
	#else
	nodeVisited[0] = 0;
	for(int i=1; i<numNodes; i++) {
		nodeVisited[i] = -1;
	}
	#endif
	
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
	
	//calculate block size (in the default case, it is chosen each iteration)
	unsigned int numBlocks;
	#ifndef DEFAULT
	numBlocks = numNodes / BLOCK_SIZE;
	if(numNodes % BLOCK_SIZE != 0) {
		numBlocks++;
	}
	#endif
	#ifndef DEFAULT
	#ifdef SHARED_MEMORY
	int paddedNumNodes = numBlocks*BLOCK_SIZE;
	#endif
	#endif
	
	//handle cuda memory
	int *devNodePtrs;
	int *devNodeNeighbors;
	#ifdef DEFAULT
	int *devCurrLevelNodes;
	int *devNextLevelNodes;
	int *devNumNextLevelNodes;
	int *devNodeVisited;
	#endif
	#ifdef NODES_OLDNEW
	int *devOldNodeVisited;
	int *devNewNodeVisited;
	bool *devDone;
	#endif
	#ifdef NODES_DISTANCES
	bool *devDone;
	int *devNodeDistances;
	#endif
	
	#ifndef DEFAULT
	cudaMemcpyToSymbol(devNumNodes, &numNodes, sizeof(int));
	#endif
	
	cudaMalloc(&devNodePtrs, sizeof(int)*(numNodes+1));
	cudaMalloc(&devNodeNeighbors, sizeof(int)*numEdges);
	#ifdef DEFAULT
	cudaMalloc(&devCurrLevelNodes, sizeof(int)*numNodes);
	cudaMalloc(&devNextLevelNodes, sizeof(int)*numNodes);
	cudaMalloc(&devNumNextLevelNodes, sizeof(int));
	cudaMalloc(&devNodeVisited, sizeof(int)*numNodes);
	#endif
	#ifdef NODES_OLDNEW
	#ifdef SHARED_MEMORY
	cudaMalloc(&devOldNodeVisited, sizeof(int)*paddedNumNodes);
	cudaMalloc(&devNewNodeVisited, sizeof(int)*paddedNumNodes);
	#else
	cudaMalloc(&devOldNodeVisited, sizeof(int)*numNodes);
	cudaMalloc(&devNewNodeVisited, sizeof(int)*numNodes);
	#endif
	cudaMalloc(&devDone, sizeof(bool));
	#endif
	#ifdef NODES_DISTANCES
	cudaMalloc(&devDone, sizeof(bool));
	#ifdef SHARED_MEMORY
	cudaMalloc(&devNodeDistances, sizeof(int)*paddedNumNodes);
	#else
	cudaMalloc(&devNodeDistances, sizeof(int)*numNodes);
	#endif
	#endif

	cudaMemcpy(devNodePtrs, nodePtrs, sizeof(int)*(numNodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodeNeighbors, nodeNeighbors, sizeof(int)*numEdges, cudaMemcpyHostToDevice);
	#ifdef DEFAULT
	cudaMemcpy(devCurrLevelNodes, currLevelNodes, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devNextLevelNodes, nextLevelNodes, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devNumNextLevelNodes, &numNextLevelNodes, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodeVisited, nodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	#endif
	#ifdef NODES_OLDNEW
	cudaMemcpy(devOldNodeVisited, nodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devNewNodeVisited, nodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(devDone, &done, sizeof(bool), cudaMemcpyHostToDevice);
	#endif
	#ifdef NODES_DISTANCES
	cudaMemcpy(devDone, &done, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodeDistances, nodeVisited, sizeof(int)*numNodes, cudaMemcpyHostToDevice);
	#endif
	
	//handle timing
	#ifdef TIMING_LOOP_TIME
	cudaEventRecord(start);
	#endif
	
	//call the cuda kernel until the traversal is finished
	done = false;
	#ifdef NODES_DISTANCES
	int level=0;
	#else
	int *temp;
	#endif
	while(!done) {
		//calculate block size and shared memory size
		#ifdef DEFAULT
		numBlocks = numCurrLevelNodes / BLOCK_SIZE;
		if(numCurrLevelNodes % BLOCK_SIZE != 0) {
			numBlocks++;
		}
		#ifdef SHARED_MEMORY
		int bq_capacity = min(SHARED_CAPACITY, numNodes)*sizeof(int);
		#endif
		#endif
		
		//transfer needed data to the device
		#ifdef DEFAULT
		numNextLevelNodes = 0;
		cudaMemcpy(devNumNextLevelNodes, &numNextLevelNodes, sizeof(int), cudaMemcpyHostToDevice);
		#else
		done = true;
		cudaMemcpy(devDone, &done, sizeof(bool), cudaMemcpyHostToDevice);
		#endif
		
		//handle timing
		#ifdef TIMING_KERNEL_TIME
		cudaEventRecord(start);
		#endif
		
		//run the kernel
		#ifdef DEFAULT
		#ifdef SHARED_MEMORY
		gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE, bq_capacity>>>(devNodePtrs, devNodeNeighbors, devNodeVisited, devCurrLevelNodes, devNextLevelNodes, numCurrLevelNodes, devNumNextLevelNodes);
		#else
		gpu_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devNodePtrs, devNodeNeighbors, devNodeVisited, devCurrLevelNodes, devNextLevelNodes, numCurrLevelNodes, devNumNextLevelNodes);
		#endif
		#endif
		#ifdef NODES_OLDNEW
		#ifdef SHARED_MEMORY
		gpu_oldnew_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devNodePtrs, devNodeNeighbors, devOldNodeVisited, devNewNodeVisited, devDone);
		#else
		gpu_oldnew_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devNodePtrs, devNodeNeighbors, devOldNodeVisited, devNewNodeVisited, devDone);
		#endif
		#endif
		#ifdef NODES_DISTANCES
		#ifdef SHARED_MEMORY
		gpu_distances_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devNodePtrs, devNodeNeighbors, devNodeDistances, devDone, level);
		#else
		gpu_distances_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(devNodePtrs, devNodeNeighbors, devNodeDistances, devDone, level);
		#endif
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
		#ifdef DEFAULT
		cudaMemcpy(&numNextLevelNodes, devNumNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost);
		numCurrLevelNodes = numNextLevelNodes;
		temp = devCurrLevelNodes;
		devCurrLevelNodes = devNextLevelNodes;
		devNextLevelNodes = temp;
		done = (numNextLevelNodes == 0);
		#endif
		#ifdef NODES_OLDNEW
		cudaMemcpy(&done, devDone, sizeof(bool), cudaMemcpyDeviceToHost);
		temp = devOldNodeVisited;
		devOldNodeVisited = devNewNodeVisited;
		devNewNodeVisited = temp;
		#endif
		#ifdef NODES_DISTANCES
		cudaMemcpy(&done, devDone, sizeof(bool), cudaMemcpyDeviceToHost);
		level++;
		#endif
	}
	
	//handle timing
	#ifdef TIMING_LOOP_TIME
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
	#endif
	
	//get the result from the device
	#ifdef DEFAULT
	cudaMemcpy(nodeVisited, devNodeVisited, sizeof(int)*numNodes, cudaMemcpyDeviceToHost);
	#endif
	#ifdef NODES_OLDNEW
	cudaMemcpy(nodeVisited, devNewNodeVisited, sizeof(int)*numNodes, cudaMemcpyDeviceToHost);
	#endif
	#ifdef NODES_DISTANCES
	cudaMemcpy(nodeVisited, devNodeDistances, sizeof(int)*numNodes, cudaMemcpyDeviceToHost);
	#endif
	
	//free device memory
	cudaFree(devNodePtrs);
	cudaFree(devNodeNeighbors);
	#ifdef DEFAULT
	cudaFree(devCurrLevelNodes);
	cudaFree(devNextLevelNodes);
	cudaFree(devNumNextLevelNodes);
	cudaFree(devNodeVisited);
	#endif
	#ifdef NODES_OLDNEW
	cudaFree(devOldNodeVisited);
	cudaFree(devNewNodeVisited);
	cudaFree(devDone);
	#endif
	#ifdef NODES_DISTANCES
	cudaFree(devDone);
	cudaFree(devNodeDistances);
	#endif
	
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
    
    //convert data structures
	#ifdef NODES_DISTANCES
	for(int i=0; i<numNodes; i++) {
		if(nodeVisited[i] == -1) {
			nodeVisited[i] = 0;
		} else {
			nodeVisited[i] = 1;
		}
	}
	#endif
	
	//free additional memory
	#ifdef DEFAULT
	free(currLevelNodes);
	free(nextLevelNodes);
	#endif
    
    #ifdef TIMING
    return time;
    #else
    return NAN;
	#endif
}
