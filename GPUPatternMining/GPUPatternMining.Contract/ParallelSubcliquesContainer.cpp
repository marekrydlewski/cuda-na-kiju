#include "ParallelSubcliquesContainer.h"

#include <algorithm>
#include <cassert>
#include <ppl.h>

ParallelSubcliquesContainer::ParallelSubcliquesContainer(unsigned short numberOfTypes)
{
	for (auto i = 0; i < numberOfTypes; ++i)
	{
		typesMap[i] = concurrency::concurrent_vector<unsigned short>();
	}

	cliquesCounter = 0;
}

void ParallelSubcliquesContainer::insertClique(std::vector<unsigned short>& clique)
{
	concurrency::critical_section mutex;

	mutex.lock();
	auto localCounter = cliquesCounter++;
	mutex.unlock();

	for (auto type : clique)
	{
		typesMap[type].push_back(localCounter);
	}
	
}

void ParallelSubcliquesContainer::insertCliques(std::vector<std::vector<unsigned short>>& cliques)
{
	for (auto& clique : cliques)
	{
		insertClique(clique);
	}
}

bool ParallelSubcliquesContainer::checkCliqueExistence(std::vector<unsigned short>& clique)
{
	assert(clique.size() >= 2);

	unsigned short currentCliquesCounter = cliquesCounter;

	std::vector<bool> types(currentCliquesCounter, false);
	std::vector<bool> typesNew(currentCliquesCounter, false);

	for (auto type : typesMap[clique[0]])
	{
		if(type < currentCliquesCounter)
			types[type] = true;
	}

	for (auto i = 1; i < clique.size(); ++i)
	{
		for (auto id : typesMap[clique[i]])
		{
			if (id < currentCliquesCounter)
			{
				if (types[id]) typesNew[id] = true;
			}
		}
		types = typesNew;
		std::fill(typesNew.begin(), typesNew.end(), false);
	}

	if (std::find(types.begin(), types.end(), true) != types.end())
		return true;

	return false;
}


ParallelSubcliquesContainer::~ParallelSubcliquesContainer()
{
}
