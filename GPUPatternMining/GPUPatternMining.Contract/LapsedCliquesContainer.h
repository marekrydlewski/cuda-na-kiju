#pragma once
#include <vector>
#include <set>
#include <algorithm>

class LapsedCliquesContainer
{
public:
	bool checkCliqueExistence(std::vector<unsigned short> clique) 
	{
		return cliques.count(clique) == 1;
	}

	void insertClique(std::vector<unsigned short> clique)
	{
		cliques.insert(clique);
	}
private:
	std::set<std::vector<unsigned short>> cliques;
};