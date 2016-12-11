#pragma once
#include "CPUMiningBaseAlgorithm.h"
#include "../../GPUPatternMining.Contract/Graph.h"
#include <map>
#include <vector>

class CPUMiningAlgorithmSeq :
	public CPUMiningBaseAlgorithm
{
private:
	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	std::vector<int> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> insTable;

	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> countUniqueInstances();

	Graph size2ColocationsGraph;
public:
	virtual void filterByDistance(float threshold) override;
	virtual void filterByPrevalence(float prevalence) override;
	virtual void createSize2ColocationsGraph();
	virtual void loadData(DataFeed* data, size_t size, unsigned int types) override;
	CPUMiningAlgorithmSeq();
	~CPUMiningAlgorithmSeq();
};

