#include "Graph.h"
#include <algorithm>


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
	this->addEdge(1, 2); //ba
	this->addEdge(2, 0); //ca
	this->addEdge(2, 3); //cd
	this->addEdge(2, 4); //bd
}

/// Matula & Beck (1983) wikipedia, linear O(n)
unsigned int Graph::getDegeneracy()
{
	std::vector<unsigned int> L;
	std::vector<std::vector<unsigned int>> D;
	int k = 0, max_dv = 0;

	//Initialize an array D such that D[i] contains a list of the vertices v that are not already in L for which dv = i.
	D.resize(tab.size());
	//Compute a number dv for each vertex v in G, the number of neighbors of v that are not already in L.
	//Initially, these numbers are just the degrees of the vertices.
	for (auto i = 0; i < tab.size(); ++i)
	{
		//Count vertexes degrees
		auto dv = std::count(tab[i].begin(), tab[i].end(), true);
		max_dv = dv > max_dv ? dv : max_dv;
		D[dv].push_back(i);
	}
	D.resize(max_dv);

	for (auto j = 0; j < tab.size(); ++j)
	{
		//Scan the array cells D[0], D[1], ... until finding an i for which D[i] is nonempty.
		auto itNonEmpty = std::find_if(D.begin(), D.end(), [](const std::vector<unsigned int>& s) { return s.size() != 0; });
		int i = (itNonEmpty - D.begin());
		//Set k to max(k, i)
		k = std::max(k, i);
		//Select a vertex v from D[i]. Add v to the beginning of L and remove it from D[i].
		L.push_back(D[k].back());
		D[k].pop_back();

	}

	return 0;
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
