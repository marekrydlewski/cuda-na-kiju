#pragma once
#include "../GPUPatternMining.Contract/Enity/DataFeed.h"

class TestDataGenerator {
public:
	DataFeed* getNoNeighboursData();
	DataFeed* getOneNeighbourRelationshipData();
	DataFeed* getLinearNeighbourRelationshipData();
	TestDataGenerator();
	~TestDataGenerator();
};