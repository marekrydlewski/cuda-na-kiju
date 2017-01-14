#include "CliqueContainer.h"



CliqueContainer::CliqueContainer()
{
}

void CliqueContainer::insertClique(std::vector<short> clique, unsigned int id)
{
	for (auto type : clique)
	{
		typesMap[type].push_back(id);
	}
}


CliqueContainer::~CliqueContainer()
{
}
