#include "catch.hpp"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmParallel.h"
#include "../GPUPatternMining.CPU.Tests/testDataGenerator.h"

TEST_CASE("NoNeighbourhoodRelationsParallel", "ParallelDistanceFilterTests")
{
	TestDataGenerator dataGenerator;
	CPUMiningAlgorithmParallel miner;

	auto graph = dataGenerator.getNoNeighboursData();
	miner.loadData(graph, 5, 5);
	miner.filterByDistance(5);

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

TEST_CASE("SingleNeighbourhoodRelationParallel", "ParallelDistanceFilterTests")
{
	TestDataGenerator dataGenerator;
	CPUMiningAlgorithmParallel miner;

	auto graph = dataGenerator.getOneNeighbourRelationshipData();
	miner.loadData(graph, 5, 5);
	miner.filterByDistance(5);

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
	REQUIRE(result[0][1][0]->size() == 1);
	REQUIRE((*result[0][1][0])[0] == 0);
}

TEST_CASE("LinearNeighbourhoodRelationParallel", "ParallelDistanceFilterTests")
{
	TestDataGenerator dataGenerator;
	CPUMiningAlgorithmParallel miner;

	auto graph = dataGenerator.getLinearNeighbourRelationshipData();
	miner.loadData(graph, 5, 5);
	miner.filterByDistance(5);

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

	REQUIRE(notEmptyTypePairs == 4);
	REQUIRE(result[0][1][0]->size() == 1);
	REQUIRE((*result[0][1][0])[0] == 0);
	REQUIRE(result[1][2][0]->size() == 1);
	REQUIRE((*result[1][2][0])[0] == 0);
	REQUIRE(result[2][3][0]->size() == 1);
	REQUIRE((*result[2][3][0])[0] == 0);
	REQUIRE(result[3][4][0]->size() == 1);
	REQUIRE((*result[3][4][0])[0] == 0);
}