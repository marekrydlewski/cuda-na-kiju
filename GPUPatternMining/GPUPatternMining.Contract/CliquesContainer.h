#pragma once

#include <map>
#include <vector>

class CliquesContainer
{
private:
	unsigned int cliquesCounter;
	std::map<short, std::vector<unsigned short>> typesMap;
public:
	CliquesContainer();
	void insertClique(std::vector<unsigned short> clique);
	bool checkCliqueExistence(std::vector<unsigned short> clique);

	virtual ~CliquesContainer();
};

