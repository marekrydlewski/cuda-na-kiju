#include "ParallelCliquesContainer.h"

#include <algorithm>
#include <cassert>
#include <ppl.h>

ParallelCliquesContainer::ParallelCliquesContainer(unsigned short numberOfTypes)
{
	for (auto i = 0; i < numberOfTypes; ++i)
	{
		typesMap[i] = concurrency::concurrent_vector<unsigned short>();
	}

	cliquesCounter = 0;
}

void ParallelCliquesContainer::insertClique(std::vector<unsigned short>& clique)
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

void ParallelCliquesContainer::insertCliques(std::vector<std::vector<unsigned short>>& cliques)
{
	for (auto& clique : cliques)
	{
		insertClique(clique);
	}
}

bool ParallelCliquesContainer::checkCliqueExistence(std::vector<unsigned short>& clique)
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


ParallelCliquesContainer::~ParallelCliquesContainer()
{
}
