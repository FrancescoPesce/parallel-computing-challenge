#include "parsing.hpp"
#ifdef COO
#include "gpu_coo.hpp"
#else
#include "gpu_csr.hpp"
#endif

#ifndef COMPARE
#ifdef CHECK_CORRECTNESS
#define COMPARE
#else
#ifdef TIMING
#define COMPARE
#endif
#endif
#endif

#ifdef COMPARE
#include "cpu.hpp"
#endif

//nvcc main.cu -o main --std c++11 -D[flags]
/*flags:
NO_SORT_INPUT
SAVE_OUTPUT
PRINT_GRAPH
CHECK_CORRECTNESS
TIMING_TOTAL_TIME (cannot be used with TIMING_LOOP_TIME, TIMING_KERNEL_TIME)
TIMING_LOOP_TIME (cannot be used with TIMING_TOTAL_TIME, TIMING_KERNEL_TIME)
TIMING_KERNEL_TIME (cannot be used with TIMING_TOTAL_TIME, TIMING_LOOP_TIME)
TIMING_FULL_LOGGING (needs TIMING_KERNEL_TIME)
SHARED_MEMORY
BLOCK_SIZE (default 512)
SHARED_CAPACITY (default 1024, only relevant with SHARED_MEMORY)
COO
NODES_OLDNEW
NODES_DISTANCES
*/

int main(int argc, char **argv) {
	#ifdef SAVE_OUTPUT
	if(argc != 3) {
		printf("usage: main input_filename output_filename\n");
		return 0;
	}
	#else
	if(argc != 2) {
		printf("usage: main input_filename\n");
		return 0;
	}
	#endif
	
	//parse the input
	int *nodePtrs;
	int *nodeNeighbors;
	int numNodes;
	int numEdges;
	createCSRGraph(argv[1], nodePtrs, nodeNeighbors, numNodes, numEdges);
	#ifdef PRINT_GRAPH
	printGraph(nodePtrs, nodeNeighbors, numNodes, numEdges);
	#endif
	#ifdef COO
	int *firstNodeEdges;
	int *secondNodeEdges;
	createCOOGraph(argv[1], firstNodeEdges, secondNodeEdges, numNodes, numEdges);
	#endif

	//traverse the graph
	int *nodeVisited = (int*) malloc(sizeof(int)*numNodes);
	#ifdef COO
	float gpu_time = coo_gpu_traversal(firstNodeEdges, secondNodeEdges, nodeVisited, numNodes, numEdges);
	#else
	float gpu_time = gpu_traversal(nodePtrs, nodeNeighbors, nodeVisited, numNodes, numEdges);
	#endif
	
	#ifdef COMPARE
	int *nodeVisitedCpu = (int*) malloc(sizeof(int)*numNodes);
	float cpu_time = cpu_traversal(nodePtrs, nodeNeighbors, nodeVisitedCpu, numNodes, numEdges);
	#endif
	#ifdef TIMING
	std::cout << "Speedup: " << cpu_time/gpu_time << "x" << std::endl;
	#endif
	#ifdef CHECK_CORRECTNESS
	compare_results(nodeVisitedCpu, nodeVisited, numNodes);
	#endif
	
	//store the output in a file
	#ifdef SAVE_OUTPUT
	printOutput(argv[2], nodeVisited, numNodes);
	#endif
	
	return 0;
}

