#script to convert from mtx to the required input format
#if a matrix that is not symmetric is used all asymmetric entries are duplicated
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_filename') 
parser.add_argument('output_filename') 
args = parser.parse_args()
input_filename = args.input_filename
output_filename = args.output_filename

#get input
with open(input_filename, "r") as f:
	data = f.readlines()

#check if the matrix is symmetric
isSymmetric = False
if "symmetric" in data[0]:
	isSymmetric = True

#get the input	
parsingHeader = True
isBInary = False
for idx, line in enumerate(data):
	if parsingHeader and line[0] != '%':
		parsingHeader = False
		split_line = line.split(" ")
		if(len(split_line) >= 3):
			numNodes = max(int(split_line[0]),int(split_line[1]))
			numEdges = int(split_line[2])
		else:
			numNodes = int(split_line[0])
			numEdges = int(split_line[1])
			isBinary = True
		nodes = [[] for _ in range(numNodes)]
	elif not parsingHeader:
		#nodes in normal mtx format are 1-based, while in binary mtx format they are 0-based
		if(not isBinary):
			node1 = int(line.split(" ")[0])-1
			node2 = int(line.split(" ")[1])-1
		else:
			node1 = int(line.split(" ")[0])
			node2 = int(line.split(" ")[1])
		nodes[node1].append(node2)

#only the upper triangular part of symmetric matrices is stored,
#for asymmetric matrices the entries can be considered the upper/lower entries of a symmetric matrix with duplicates
#duplicates are removed
if not isSymmetric:
	for node1, neigh_list in enumerate(nodes):
		for node2 in neigh_list:
			if node1 != node2 and node1 in nodes[node2]:
				nodes[node2].remove(node1)
				numEdges -= 1
	
#write parsed data			
with open(output_filename, "w") as f:
	f.write(str(numNodes)+" "+str(numEdges)+"\n")
	for node1, neigh_list in enumerate(nodes):
		for node2 in neigh_list:
			f.write(str(node1)+" "+str(node2)+"\n")
