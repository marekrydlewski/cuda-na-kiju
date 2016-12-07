#include "Graph.h"



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
