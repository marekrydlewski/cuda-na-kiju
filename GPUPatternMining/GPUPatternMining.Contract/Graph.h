#pragma once
#include <vector>
#include <set>
struct Graph
{
	Graph(size_t size);

	//adjacency matrix
	std::vector<std::vector<bool>> tab;

	void setSize(size_t size);
	void addEdge(unsigned int v1, unsigned int v2);
	void getMock();

	std::vector<unsigned int> getVertexNeighboursOfHigherIndex(unsigned int v);
	std::vector<unsigned int> getVertexNeighboursOfLowerIndex(unsigned int v);
	std::vector<unsigned int> getVertexNeighbours(unsigned int v);

	std::pair<unsigned int, std::vector<unsigned int>> getDegeneracy();

	unsigned int tomitaMaximalPivot(const std::vector<unsigned int>& SUBG, const std::vector<unsigned int>& CAND);
	std::vector<std::vector<unsigned int>> bkPivot(std::vector<unsigned int> M, std::vector<unsigned int> K, std::vector<unsigned int> T);

	Graph();
	~Graph();
};

