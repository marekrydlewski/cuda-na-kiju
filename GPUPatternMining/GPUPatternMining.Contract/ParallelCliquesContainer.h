#pragma once

#include <vector>
#include <algorithm>
#include <ppl.h>
#include <concurrent_unordered_set.h>

#include "Hashers.h"

class ParallelCliquesContainer
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

	ParallelCliquesContainer() {};
	~ParallelCliquesContainer() {};

private:
	concurrency::concurrent_unordered_set<std::vector<unsigned short>, vector_hash> cliques;
};