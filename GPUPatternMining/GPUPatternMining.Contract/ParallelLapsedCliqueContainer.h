#pragma once

#include <vector>
#include <concurrent_unordered_set.h>

class ParallelLapsedCliqueContainer
{
public:
	bool checkCliqueExistence(std::vector<unsigned short>& clique)
	{
		return cliques.count(clique) == 1;
	}

	void insertClique(std::vector<unsigned short>& clique)
	{
		cliques.insert(clique);
	}
private:
	concurrency::concurrent_unordered_set<std::vector<unsigned short>> cliques;
};

