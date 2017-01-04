#pragma once
#include "CPUMiningBaseAlgorithm.h"
#include "../../GPUPatternMining.Contract/Graph.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include <map>
#include <vector>

class CPUMiningAlgorithmParallel :
	public CPUMiningBaseAlgorithm
{
public:
	CPUMiningAlgorithmParallel();
	virtual ~CPUMiningAlgorithmParallel();

	void loadData(
		DataFeed* data,
		size_t size,
		unsigned int types) override;

	void filterByDistance(float threshold) override;

private:
	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	std::vector<int> typeIncidenceCounter;
};