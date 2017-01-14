#pragma once

#include<map>
#include<vector>

class CliqueContainer
{
private:
	std::map<short, std::vector<unsigned short>> typesMap;
public:
	CliqueContainer();
	void insertClique(std::vector<short> clique, unsigned int id);
	bool checkCliqueExistence(std::vector<short> clique);

	virtual ~CliqueContainer();
};

