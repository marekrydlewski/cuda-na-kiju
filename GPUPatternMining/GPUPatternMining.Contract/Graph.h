#pragma once
#include <vector>
#include <set>
struct Graph
{
	//adjacency matrix
	std::vector<std::vector<bool>> tab;

	void setSize(size_t size);
	void addEdge(unsigned int v1, unsigned int v2);
	void getMock();

	std::set<unsigned int> getVertexNeighboursOfHigherIndex(unsigned int v);
	std::set<unsigned int> getVertexNeighboursOfLowerIndex(unsigned int v);
	std::set<unsigned int> getVertexNeighbours(unsigned int v);
	std::pair<unsigned int, std::vector<unsigned int>> getDegeneracy();
	unsigned int tomitaMaximalPivot(const std::set<unsigned int>& SUBG, const std::set<unsigned int>& CAND);
	std::set<std::vector<unsigned int>> bkPivot(std::set<unsigned int> M, std::set<unsigned int> K, std::set<unsigned int> T);

	Graph(size_t size);
	Graph();
	~Graph();
};

