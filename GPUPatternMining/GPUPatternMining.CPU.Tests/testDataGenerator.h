#pragma once
#include "../GPUPatternMining.Contract/Enity/DataFeed.h"

class TestDataGenerator {
public:
	DataFeed* getNoNeighboursData();
	DataFeed* getOneNeighbourRelationshipData();
	DataFeed* getLinearNeighbourRelationshipData();
	DataFeed* getDataForPrevalenceTests();
	TestDataGenerator();
	~TestDataGenerator();
};