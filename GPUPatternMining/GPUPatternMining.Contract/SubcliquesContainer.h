#pragma once

#include <unordered_map>
#include <vector>

class SubcliquesContainer
{
private:
	unsigned int cliquesCounter;
	std::unordered_map<short, std::vector<unsigned short>> typesMap;
public:
	SubcliquesContainer();
	void insertClique(std::vector<unsigned short>& clique);
	void insertCliques(std::vector<std::vector<unsigned short>>& cliques);
	bool checkCliqueExistence(std::vector<unsigned short>& clique);

	virtual ~SubcliquesContainer();
};

