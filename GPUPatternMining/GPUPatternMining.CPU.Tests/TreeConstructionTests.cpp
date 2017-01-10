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

TEST_CASE("ColocationsWhereSomeSize2ColocationsWereNotPrevalent", "TreeConstructionTest")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	int threshold = 5;
	float prevalence = 0.4;

	auto data = generator.getDataForMixedPrevalenceResults();
	miner.loadData(data, 6, 3);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.filterMaximalCliques(prevalence);
	REQUIRE(result.size() == 2);
}

TEST_CASE("NonPrevalentColocations", "TreeConstructionTest")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	float threshold = 5.1;
	float prevalence = 0.3;

	auto data = generator.getDataForTreeTest();
	miner.loadData(data, 14, 5);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.filterMaximalCliques(prevalence);
	REQUIRE(result.size() == 4);
}