#pragma once
#include <tuple>

#include "DataLoader.h"
#include "Enity\DataFeed.h"

enum class DataSet { Fast, Medium, VeryLarge };

class SimulatedRealDataProvider
{
private:
	DataLoader loader;
	static const std::map<DataSet, std::string> datasetNames;
public:
	std::tuple<DataFeed*, int, int> getTestData(DataSet dataset = DataSet::Fast);
};


