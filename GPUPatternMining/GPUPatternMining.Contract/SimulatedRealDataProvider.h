#pragma once
#include <tuple>

#include "DataLoader.h"
#include "Enity\DataFeed.h"

enum class DataSet { VeryFast, Fast, Medium, Huge };
enum class InstanceSizeTestDataSet { Level1, Level2, Level3, Level4, Level5, Level6, Level7, Level8 };

class SimulatedRealDataProvider
{
private:
	DataLoader loader;
	static const std::map<DataSet, std::string> datasetNames;
	static const std::map<InstanceSizeTestDataSet, std::string> instanceSizeTestDatasetNames;
public:
	std::tuple<DataFeed*, int, int> getTestData(DataSet dataset = DataSet::Fast);
	std::tuple<DataFeed*, int, int> getInstanceSizeTestData(InstanceSizeTestDataSet dataset = InstanceSizeTestDataSet::Level1);
	std::tuple<DataFeed*, int, int> getTestData(std::string fileName);
};


