#pragma once
#include <vector>
#include <set>
struct Graph
{
	Graph(size_t size);

	//adjacency matrix
	std::vector<std::vector<bool>> tab;

	void setSize(size_t size);
	void addEdge(unsigned short v1, unsigned short v2);
	void getMock();

	std::vector<unsigned short> getVertexNeighboursOfHigherIndex(unsigned short v);
	std::vector<unsigned short> getVertexNeighboursOfLowerIndex(unsigned short v);
	std::vector<unsigned short> getVertexNeighbours(unsigned short v);

	std::pair<unsigned short, std::vector<unsigned short>> getDegeneracy();

	unsigned short tomitaMaximalPivot(const std::vector<unsigned short>& SUBG, const std::vector<unsigned short>& CAND);
	std::vector<std::vector<unsigned short>> bkPivot(std::vector<unsigned short> M, std::vector<unsigned short> K, std::vector<unsigned short> T);

	Graph();
	~Graph();
};

