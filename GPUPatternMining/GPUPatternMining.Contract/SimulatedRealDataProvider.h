#pragma once
#include <tuple>

#include "DataLoader.h"
#include "Enity\DataFeed.h"

class SimulatedRealDataProvider
{
private:
	DataLoader loader;
public:
	std::tuple<DataFeed*, int, int> getTestData(std::string fileName);
};


