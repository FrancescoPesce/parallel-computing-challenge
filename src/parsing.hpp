#include <stdio.h>
#include <malloc.h>
#include <vector>
#ifndef NO_SORT_INPUT
#include <algorithm>
#endif

//input file structure:
//line 1: numNodes numEdges <- number of edges specified in the file, not number of edges in total
//lines 2-numEdges+1: node1 node2 <- lines in any order, but no duplicates and if "node1 node2" with node1 != node2 is a line, then "node2 node1" cannot be a line
//if SORT_INPUT is not defined, then lines have to be ordered by the first node and the by the second node

void parseFile(const char* filename, std::vector<std::vector<int>> &edges, int &numNodes);
void preParseEdges(const char* filename, int &numNodes, int &numEdges, std::vector<std::vector<int>> &edges);

void createCSRGraph(const char* filename, int* &nodePtrs, int* &nodeNeighbors, int &numNodes, int &numEdges) {
	//parse the file
	std::vector<std::vector<int>> edges;
	preParseEdges(filename, numNodes, numEdges, edges);
	
	//allocate memory
	nodePtrs = (int*) malloc(sizeof(int)*(numNodes+1));
	nodeNeighbors = (int*) malloc(sizeof(int)*numEdges);
	
	//create csr representation
	int idx = 0;
	for(int i=0; i<numNodes; i++) {
		nodePtrs[i] = idx;
		for(int j=0; j<edges[i].size(); j++) {
			nodeNeighbors[idx] = edges[i][j];
			++idx;
		}
	}
	nodePtrs[numNodes] = numEdges;
}

void createCOOGraph(const char* filename, int* &firstNodeEdges, int* &secondNodeEdges, int &numNodes, int &numEdges) {
	//parse the file
	std::vector<std::vector<int>> edges;
	preParseEdges(filename, numNodes, numEdges, edges);
	
	//allocate memory
	firstNodeEdges = (int*) malloc(sizeof(int)*numEdges);
	secondNodeEdges = (int*) malloc(sizeof(int)*numEdges);
	
	//create coo representation
	int idx = 0;
	for(int i=0; i<numNodes; i++) {
		for(int j=0; j<edges[i].size(); j++) {
			firstNodeEdges[idx] = i;
			secondNodeEdges[idx] = edges[i][j];
			++idx;
		}
	}
}

//duplicate edges and remove auto-edges
void preParseEdges(const char* filename, int &numNodes, int &numEdges, std::vector<std::vector<int>> &edges) {
	//parse the file
	std::vector<std::vector<int>> old_edges;
	parseFile(filename, old_edges, numNodes);
	
	//duplicate edges and remove auto-edges
	for(int i=0; i<numNodes; i++) {
		edges.emplace_back(std::vector<int>());
	}
	numEdges = 0;
	for(int i=0; i<numNodes; i++) {
		for(int j=0; j<old_edges[i].size(); j++) {
			int node2 = old_edges[i][j];
			if(i != node2) {
				edges[i].emplace_back(node2);
				edges[node2].emplace_back(i);
				numEdges += 2;
			}
		}
	}
	
	#ifndef NO_SORT_INPUT
	//sort edges
	for(int i=0; i<numNodes; i++) {
		std::sort(edges[i].begin(), edges[i].end());
	}
	#endif
}

//get graph information
void parseFile(const char* filename, std::vector<std::vector<int>> &edges, int &numNodes) {
	//open file
	FILE* f = fopen(filename, "r");
	
	//get number of nodes and edges
	int numEdges;
	fscanf(f, "%d", &numNodes);
	fscanf(f, "%d", &numEdges);
	
	//get edges information
	int node1;
	int node2;
	for(int i=0; i<numNodes; i++) {
		edges.emplace_back(std::vector<int>());
	}
	for(int i=0; i<numEdges; i++) {
		fscanf(f, "%d", &node1);
		fscanf(f, "%d", &node2);
		edges[node1].emplace_back(node2);
	}
	
	//close file
	fclose(f);
}

//a basic utility to check if the parsing worked
void printGraph(const int* nodePtrs, const int* nodeNeighbors, int numNodes, int numEdges) {
	for(int i=0; i<numNodes; i++) {
		printf("%d: ", i);
		for(int j=nodePtrs[i]; j<numEdges && j<nodePtrs[i+1]; j++) {
			int neigh = nodeNeighbors[j];
			printf("%d ", neigh);
		}
		printf("\n");
	}
	printf("\n");
	fflush(stdout);
}

//write which nodes have been visited in a file, nodes are in order and separated by a space
void printOutput(const char* filename, int* nodeVisited, int numNodes) {
	//open file
	FILE* f = fopen(filename, "w");
	
	//print visiting information
	for(int i=0; i<numNodes; i++) {
		if(nodeVisited[i]) {
			fprintf(f,"%d ",i);
		}
	}
	
	//close file
	fclose(f);
}
