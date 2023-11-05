#include <malloc.h>
#include <vector>
#include <iostream>
#ifdef TIMING
#include <chrono>
#endif

//executes one iteration of graph traversal
void cpu_queueing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited, int *currLevelNodes, int *nextLevelNodes, const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {
	for(int idx=0; idx<numCurrLevelNodes; idx++) {
		int node = currLevelNodes[idx];
		
		for(int nbrIdx=nodePtrs[node]; nbrIdx<nodePtrs[node+1]; ++nbrIdx) {
			int neighbor = nodeNeighbors[nbrIdx];
			if(!nodeVisited[neighbor]) {
				nodeVisited[neighbor] = 1;
				nextLevelNodes[*numNextLevelNodes] = neighbor;
				++(*numNextLevelNodes);
			}
		}
	}
}

//executes full graph traversal from node 0, returns time
float cpu_traversal(int *nodePtrs, int *nodeNeighbors, int *nodeVisited, int numNodes, int numEdges) {	
	//allocate space
	int *currLevelNodes = (int*) malloc(sizeof(int)*numNodes);
	int *nextLevelNodes = (int*) malloc(sizeof(int)*numNodes);
	
	//traversal starts at node 0, initialize data structures
	nodeVisited[0] = 1;
	for(int i=1; i<numNodes; i++) {
		nodeVisited[i] = 0;
	}
	currLevelNodes[0] = 0;
	int numCurrLevelNodes = 1;
	int numNextLevelNodes = -1;
	
	//call queueing function until the queue is empty
	#ifdef TIMING
	const auto t0 = std::chrono::high_resolution_clock::now();
	#endif
	int *temp;
	while(numNextLevelNodes != 0) {
		numNextLevelNodes = 0;
		cpu_queueing(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes, numCurrLevelNodes, &numNextLevelNodes);
		numCurrLevelNodes = numNextLevelNodes;
		temp = currLevelNodes;
		currLevelNodes = nextLevelNodes;
		nextLevelNodes = temp;
	}
	#ifdef TIMING
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();  
    float time = dt;
    time /= 1e+6;
    std::cout << "CPU time:                       " << time << " ms" << std::endl;
    #endif
    
    //free data structures
	free(currLevelNodes);
	free(nextLevelNodes);
    
    #ifdef TIMING
    return time;
    #else
    return 0;
	#endif
}

//compares cpu and gpu results
void compare_results(int *nodeVisitedGolden, int *nodeVisited, int numNodes) {
	std::vector<int> missing{};
	std::vector<int> unneeded{};
	for(int i=0; i<numNodes; i++) {
		if(nodeVisitedGolden[i] && !nodeVisited[i]) {
			missing.emplace_back(i);
		}
		else if(!nodeVisitedGolden[i] && nodeVisited[i]) {
			unneeded.emplace_back(i);
		}
	}
	if(missing.size() != 0) {
		std::cout << "Missing nodes: ";
		for (int i=0; i<missing.size(); i++) {
			std::cout << missing[i] << " ";
		}
		std::cout << std::endl;
	}
	if(unneeded.size() != 0) {
		std::cout << "Unneeded nodes: ";
		for (int i=0; i<unneeded.size(); i++) {
			std::cout << unneeded[i] << " ";
		}
		std::cout << std::endl;
	}
}
