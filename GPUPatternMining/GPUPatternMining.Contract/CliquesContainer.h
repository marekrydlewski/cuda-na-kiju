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
	void insertClique(std::vector<short> clique, unsigned int id);
	bool checkCliqueExistence(std::vector<short> clique);

	virtual ~CliquesContainer();
};

