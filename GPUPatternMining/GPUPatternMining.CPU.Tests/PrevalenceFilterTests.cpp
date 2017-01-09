#include "catch.hpp"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU.Tests/testDataGenerator.h"

TEST_CASE("PrevalenceTooLow", "PrevalenceFilterTests")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	DataFeed* data = generator.getDataForPrevalenceTests();
	int threshold = 5;
	float prevalence = 0.7;

	miner.loadData(data, 6, 2);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	auto result = miner.getInsTable();

	int notEmptyTypePairs = 0;

	for (auto& row : result)
	{
		for (auto& column : row.second)
		{
			for (auto& instance : column.second)
			{
				if (instance.second->size() > 0)
				{
					++notEmptyTypePairs;
				}
			}
		}
	}

	REQUIRE(notEmptyTypePairs == 0);
}

TEST_CASE("PrevalenceHighEnough", "PrevalenceFilterTests")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	DataFeed* data = generator.getDataForPrevalenceTests();
	int threshold = 5;
	float prevalence = 0.45; //actual prevalence should be 0.5 - 2 colocations, 4 type0 instances, 2 type1 instances

	miner.loadData(data, 6, 2);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	auto result = miner.getInsTable();

	int notEmptyTypePairs = 0;

	for (auto& row : result)
	{
		for (auto& column : row.second)
		{
			for (auto& instance : column.second)
			{
				if (instance.second->size() > 0)
				{
					++notEmptyTypePairs;
				}
			}
		}
	}

	REQUIRE(notEmptyTypePairs == 2);
	REQUIRE((*result[0][1][0])[0] == 0);
	REQUIRE((*result[0][1][1])[0] == 1);
}

TEST_CASE("MixedPrevalenceResults", "PrevalenceFilterTests")
{
	TestDataGenerator generator;
	CPUMiningAlgorithmSeq miner;
	DataFeed* data = generator.getDataForMixedPrevalenceResults();
	int threshold = 5;
	float prevalence = 0.4;

	miner.loadData(data, 6, 3);
	miner.filterByDistance(threshold);
	miner.filterByPrevalence(prevalence);
	auto result = miner.getInsTable();

	int notEmptyTypePairs = 0;

	for (auto& row : result)
	{
		for (auto& column : row.second)
		{
			for (auto& instance : column.second)
			{
				if (instance.second->size() > 0)
				{
					++notEmptyTypePairs;
				}
			}
		}
	}

	REQUIRE(notEmptyTypePairs == 1);
	REQUIRE((*result[1][2][0]).size() == 1);
	REQUIRE((*result[1][2][0])[0] == 0);
}