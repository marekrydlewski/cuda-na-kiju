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
	void filterByPrevalence(float prevalence) override;
	void constructMaximalCliques() override;

	std::vector<std::vector<unsigned int>> bkPivot(std::vector<unsigned int> M, std::vector<unsigned int> K, std::vector<unsigned int> T);

private:
	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	std::vector<int> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> insTable;

	Graph size2ColocationsGraph;
};