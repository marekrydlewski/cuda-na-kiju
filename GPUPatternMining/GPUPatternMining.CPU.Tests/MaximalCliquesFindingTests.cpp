#include "catch.hpp"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU.Tests/testDataGenerator.h"

TEST_CASE("2ElementClique", "MaximalCliquesFindingTests")
{
	CPUMiningAlgorithmSeq miner;
	TestDataGenerator generator;
	int threshold = 5;
	float prevalence = 0.5;
	auto data = generator.getDataForMaximalCliqueSize2();

	miner.loadData(data, 2, 2);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.getMaximalCliques();

	REQUIRE(result.size() == 1);
	REQUIRE(result[0].size() == 2);
}

TEST_CASE("1ElementCliques", "MaximalCliquesFindingTests")
{
	CPUMiningAlgorithmSeq miner;
	TestDataGenerator generator;
	int threshold = 5;
	float prevalence = 0.5;
	auto data = generator.getDataForMaximalCliqueSize1();

	miner.loadData(data, 2, 2);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.getMaximalCliques();

	REQUIRE(result.size() == 2);
	REQUIRE(result[0].size() == 1);
	REQUIRE(result[1].size() == 1);
}

TEST_CASE("4ElementClique", "MaximalCliquesFindingTests")
{
	CPUMiningAlgorithmSeq miner;
	TestDataGenerator generator;
	int threshold = 5;
	float prevalence = 0.5;
	auto data = generator.getDataForMaximalCliqueSize4();

	miner.loadData(data, 4, 4);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	miner.constructMaximalCliques();

	auto result = miner.getMaximalCliques();

	REQUIRE(result.size() == 1);
	REQUIRE(result[0].size() == 4);
}