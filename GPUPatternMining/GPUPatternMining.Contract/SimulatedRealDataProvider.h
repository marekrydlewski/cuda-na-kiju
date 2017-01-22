#pragma once
#include <tuple>

#include "DataLoader.h"
#include "Enity\DataFeed.h"

enum class DataSet { VeryFast, Fast, Medium, Huge };

class SimulatedRealDataProvider
{
private:
	DataLoader loader;
	static const std::map<DataSet, std::string> datasetNames;
public:
	std::tuple<DataFeed*, int, int> getTestData(DataSet dataset = DataSet::Fast);
};


