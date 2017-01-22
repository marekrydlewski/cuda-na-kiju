#include "SubcliquesContainer.h"

#include <algorithm>
#include <cassert>

SubcliquesContainer::SubcliquesContainer()
{
	cliquesCounter = 0;
}

void SubcliquesContainer::insertClique(std::vector<unsigned short>& clique)
{
	for (auto type	: clique)
	{
		typesMap[type].push_back(cliquesCounter);
	}
	++cliquesCounter;
}

void SubcliquesContainer::insertCliques(std::vector<std::vector<unsigned short>>& cliques)
{
	for (auto& clique : cliques)
	{
		insertClique(clique);
	}
}

bool SubcliquesContainer::checkCliqueExistence(std::vector<unsigned short>& clique)
{
	assert(clique.size() >= 2);

	std::vector<bool> types(cliquesCounter, false);
	std::vector<bool> typesNew(cliquesCounter, false);

	for (auto type : typesMap[clique[0]])
	{
		types[type] = true;
	}

	for (auto i = 1; i < clique.size(); ++i)
	{
		for (auto id : typesMap[clique[i]])
		{
			if (types[id]) typesNew[id] = true;
		}
		types = typesNew;
		std::fill(typesNew.begin(), typesNew.end(), false);
	}

	if (std::find(types.begin(), types.end(), true) != types.end())
		return true;

	return false;
}


SubcliquesContainer::~SubcliquesContainer()
{
}
