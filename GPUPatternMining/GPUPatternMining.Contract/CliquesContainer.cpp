#include "CliquesContainer.h"

#include <algorithm>
#include <cassert>

CliquesContainer::CliquesContainer()
{
	cliquesCounter = 0;
}

void CliquesContainer::insertClique(std::vector<short> clique, unsigned int id)
{
	for (auto type : clique)
	{
		typesMap[type].push_back(id);
	}
	++cliquesCounter;
}

bool CliquesContainer::checkCliqueExistence(std::vector<short> clique)
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

	if (std::find(types.begin(), types.end(), false) != types.end())
		return true;

	return false;
}


CliquesContainer::~CliquesContainer()
{
}
