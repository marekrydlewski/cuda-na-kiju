#pragma 
#include <vector>
struct Graph
{
	//adjacency matrix
	std::vector<std::vector<bool>> tab;
	void setSize(size_t size);
	void addEdge(unsigned int v1, unsigned int v2);
	void getMock();
	std::pair<unsigned int, std::vector<unsigned int>> getDegeneracy();
	Graph(size_t size);
	Graph();
	~Graph();
};

