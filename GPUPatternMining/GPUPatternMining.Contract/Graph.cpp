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

	D.resize(tab.size());
	for (auto i = 0; i < tab.size(); ++i)
	{
		//count vertexes degrees
		auto dv = std::count(tab[i].begin(), tab[i].end(), true);
		max_dv = dv > max_dv ? dv : max_dv;
		D[dv].push_back(i);
	}
	D.resize(max_dv);

	for (auto j = 0; j < tab.size(); ++j)
	{
		auto itNonEmpty = std::find_if(D.begin(), D.end(), [](const std::vector<unsigned int>& s) { return s.size() != 0; });
		int i = (itNonEmpty - D.begin());
		k = std::max(k, i);
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
