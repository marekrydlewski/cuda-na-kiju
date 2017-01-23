#pragma once

#include <vector>
#include <set>
#include <algorithm>

class CliquesContainer
{
public:
	bool checkCliqueExistence(std::vector<unsigned short>& clique)
	{
		return cliques.count(clique) == 1;
	}

	bool checkSubcliqueExistence(std::vector<unsigned short>& clique)
	{
		bool isSubclique;
		for (auto& c : cliques)
		{
			if (clique.size() < c.size()) continue;

			auto it = clique.begin();
			isSubclique = true;
			for (auto id : c)
			{
				it = std::find(it, clique.end(), id);
				if (it == clique.end()) {
					isSubclique = false;
					break;
				}
			}
			if (isSubclique) return true;
		}
		return false;
	}

	void insertClique(std::vector<unsigned short>& clique)
	{
		cliques.insert(clique);
	}

	void insertCliques(std::vector<std::vector<unsigned short>>& cliques)
	{
		this->cliques.insert(cliques.begin(), cliques.end());
	}

	CliquesContainer() {};
	~CliquesContainer() {};
private:
	std::set<std::vector<unsigned short>> cliques;
};