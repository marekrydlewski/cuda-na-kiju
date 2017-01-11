#include "Graph.h"
#include <algorithm>
#include <set>
#include <vector>
#include <iterator>


void Graph::setSize(size_t size)
{
	tab.resize(size);
	for (unsigned int i = 0; i < size; i++)
	{
		tab[i].resize(size);
	}
}

void Graph::addEdge(unsigned int v1, unsigned int v2)
{
	tab[v1][v2] = true;
	tab[v2][v1] = true;
}

std::vector<unsigned int> Graph::getVertexNeighboursOfHigherIndex(unsigned int v)
{
	std::vector<unsigned int> neighbours;

	for (unsigned int v2 = v + 1; v2 < tab.size(); ++v2)
	{
		if (tab[v][v2] || tab[v2][v])
		{
			neighbours.push_back(v2);
		}
	}

	return neighbours;
}

std::vector<unsigned int> Graph::getVertexNeighboursOfLowerIndex(unsigned int v)
{
	std::vector<unsigned int> neighbours;

	for (int v2 = v - 1; v2 >= 0; --v2)
	{
		if (tab[v][v2] || tab[v2][v])
		{
			neighbours.push_back(v2);
		}
	}

	return neighbours;
}

std::vector<unsigned int> Graph::getVertexNeighbours(unsigned int v)
{
	auto neighbours = getVertexNeighboursOfHigherIndex(v);
	auto neighbours2 = getVertexNeighboursOfLowerIndex(v);

	neighbours.insert(neighbours.begin(), neighbours2.begin(), neighbours2.end());
	return neighbours;
}

///China example Fig. 4
void Graph::getMock()
{
	for (unsigned int i = 0; i < tab.size(); i++)
	{
		tab[i].clear();
	}
	tab.clear();

	this->setSize(5);

	this->addEdge(0, 1); //ba
	this->addEdge(0, 3); //da
	this->addEdge(0, 4); //ea
	this->addEdge(1, 2); //bc
	this->addEdge(2, 0); //ca
	this->addEdge(2, 3); //cd
	this->addEdge(1, 3); //bd
	this->addEdge(4, 2); //ce
}

/// Matula & Beck (1983) wikipedia, linear O(n)
std::pair<unsigned int, std::vector<unsigned int>> Graph::getDegeneracy()
{
	std::vector<unsigned int> L;
	std::vector<std::vector<unsigned int>> D;
	int k = 0;

	//Initialize an array D such that D[i] contains a list of the vertices v that are not already in L for which dv = i.
	D.resize(tab.size());
	//Compute a number dv for each vertex v in G, the number of neighbors of v that are not already in L.
	//Initially, these numbers are just the degrees of the vertices.
	for (auto i = 0; i < tab.size(); ++i)
	{
		//Count vertexes degrees
		auto dv = std::count(tab[i].begin(), tab[i].end(), true);
		D[dv].push_back(i);
	}
	//Resizing - removes last empties
	while (D.back().size() == 0)
		D.pop_back();

	for (auto j = 0; j < tab.size(); ++j)
	{
		//Scan the array cells D[0], D[1], ... until finding an i for which D[i] is nonempty.
		auto itNonEmpty = std::find_if(D.begin(), D.end(),
			[](const std::vector<unsigned int>& s) { return s.size() != 0; });
		int i = (itNonEmpty - D.begin());
		//Set k to max(k, i)
		k = std::max(k, i);
		//Select a vertex v from D[i]. Add v to the beginning of L and remove it from D[i].
		auto v = D[i].back();
		L.push_back(v);
		D[i].pop_back();
		//neighbors
		for (auto vi = 0; vi < tab[v].size(); ++vi)
		{
			//For each neighbor w of v not already in L
			if (tab[v][vi] && std::find(L.begin(), L.end(), vi) == L.end())
			{
				//Find location of vi in D
				for (auto dRowIndex = 0; dRowIndex < D.size(); ++dRowIndex)
				{
					auto neighborFound = std::find_if(D[dRowIndex].begin(), D[dRowIndex].end(),
						[&vi](const unsigned int& dvalue) { return dvalue == vi; });

					//Subtract one from dw and move w to the cell of D corresponding to the new value of dw.
					while (neighborFound != D[dRowIndex].end())
					{
						unsigned int neighborValue = *neighborFound;
						D[dRowIndex].erase(neighborFound);
						D[dRowIndex - 1].push_back(neighborValue);

						neighborFound = std::find_if(D[dRowIndex].begin(), D[dRowIndex].end(),
							[&vi](const unsigned int& dvalue) { return dvalue == vi; });
					}
				}
			}
		}
	}
	//k = degeneracy of graph, L = list in degeneracy ordering
	return std::make_pair(k, L);
}

//vectors
///Tomita Tanaka 2006 maximal pivot algorithm
unsigned int Graph::tomitaMaximalPivot(const std::vector<unsigned int>& SUBG, const std::vector<unsigned int>& CAND)
{
	unsigned int u, maxCardinality = 0;
	for (auto s : SUBG)
	{
		auto neighbors = getVertexNeighbours(s);
		std::sort(neighbors.begin(), neighbors.end());
		
		std::vector<unsigned int> nCANDunion(neighbors.size() + CAND.size());

		auto itUnion = std::set_union(CAND.begin(), CAND.end(), neighbors.begin(), neighbors.end(), nCANDunion.begin());
		nCANDunion.resize(itUnion - nCANDunion.begin());

		if (nCANDunion.size() >= maxCardinality)
		{
			u = s;
			maxCardinality = nCANDunion.size();
		}
	}
	return u;
}

//vectors
std::vector<std::vector<unsigned int>> Graph::bkPivot(
	std::vector<unsigned int> M,
	std::vector<unsigned int> K,
	std::vector<unsigned int> T)
{
	std::vector<std::vector<unsigned int>> maximalCliques;
	std::vector<unsigned int> MTunion(M.size() + T.size());
	std::vector<unsigned int> MpivotNeighboursDifference(M.size());

	std::vector<unsigned int>::iterator it;

	std::sort(M.begin(), M.end());
	std::sort(T.begin(), T.end());
	std::sort(K.begin(), K.end());

	it = std::set_union(
		M.begin(),
		M.end(),
		T.begin(),
		T.end(),
		MTunion.begin());

	MTunion.resize(it - MTunion.begin());

	if (MTunion.size() == 0)
	{
		maximalCliques.push_back(K);
		return maximalCliques;
	}

	unsigned int pivot = tomitaMaximalPivot(MTunion, M);

	auto pivotNeighbours = getVertexNeighbours(pivot);
	std::sort(pivotNeighbours.begin(), pivotNeighbours.end());

	it = std::set_difference(
		M.begin(),
		M.end(),
		pivotNeighbours.begin(),
		pivotNeighbours.end(),
		MpivotNeighboursDifference.begin());

	MpivotNeighboursDifference.resize(it - MpivotNeighboursDifference.begin());

	for (auto const vertex : MpivotNeighboursDifference)
	{
		auto c = getVertexNeighbours(vertex);

		std::vector<unsigned int> vertexNeighbours(c.begin(), c.end());
		std::vector<unsigned int> vertexVector = { vertex };
		std::vector<unsigned int> KvertexUnion(K.size() + 1);
		std::vector<unsigned int> MvertexNeighboursIntersection(M.size());
		std::vector<unsigned int> TvertexNeighboursIntersection(T.size());

		std::sort(vertexNeighbours.begin(), vertexNeighbours.end());

		std::set_union(
			K.begin(),
			K.end(),
			vertexVector.begin(),
			vertexVector.end(),
			KvertexUnion.begin());

		it = std::set_intersection(
			M.begin(),
			M.end(),
			vertexNeighbours.begin(),
			vertexNeighbours.end(),
			MvertexNeighboursIntersection.begin());

		MvertexNeighboursIntersection.resize(it - MvertexNeighboursIntersection.begin());

		it = std::set_intersection(
			T.begin(),
			T.end(),
			vertexNeighbours.begin(),
			vertexNeighbours.end(),
			TvertexNeighboursIntersection.begin());

		TvertexNeighboursIntersection.resize(it - TvertexNeighboursIntersection.begin());

		auto generatedCliques = bkPivot(
			MvertexNeighboursIntersection,
			KvertexUnion,
			TvertexNeighboursIntersection);

		maximalCliques.insert(maximalCliques.end(), generatedCliques.begin(), generatedCliques.end());
	}
	return maximalCliques;
}

Graph::Graph(size_t size)
{
	setSize(size);
}


Graph::Graph()
{
}


Graph::~Graph()
{
}
