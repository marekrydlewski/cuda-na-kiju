#pragma once

#include <map>
#include <vector>

class ParallelCliquesContainer
{
private:
	unsigned int cliquesCounter;
	std::map<short, std::vector<unsigned short>> typesMap;
public:
	ParallelCliquesContainer();
	void insertClique(std::vector<unsigned short>& clique);
	void insertCliques(std::vector<std::vector<unsigned short>>& cliques);
	bool checkCliqueExistence(std::vector<unsigned short>& clique);

	virtual ~ParallelCliquesContainer();
};

