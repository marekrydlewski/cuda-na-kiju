#include "catch.hpp"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU.Tests/testDataGenerator.h"

TEST_CASE("2DistinctColocations", "TreeConstructionTests")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	int threshold = 5;
	float prevalence = 0.1;

	auto data = generator.getDataForDistinctCliques();
	miner.loadData(data, 5, 5);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.filterMaximalCliques(prevalence);

	REQUIRE(result.size() == 2);
}

TEST_CASE("2OverlappingColocations", "TreeConstructionTests")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	int threshold = 5;
	float prevalence = 0.1;

	auto data = generator.getDataFor2OverlappingCliques();
	miner.loadData(data, 5, 5);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.filterMaximalCliques(prevalence);

	REQUIRE(result.size() == 2);
}