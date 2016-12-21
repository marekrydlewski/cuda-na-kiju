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
	std::vector<std::vector<unsigned int>> maximalCliques;
	Graph size2ColocationsGraph;

	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> countUniqueInstances();
	std::vector<std::vector<unsigned int>> bkPivot(std::vector<unsigned int> M, std::vector<unsigned int> K, std::vector<unsigned int> T);
	unsigned int tomitaMaximalPivot(const std::vector<unsigned int>& SUBG, const std::vector<unsigned int>& CAND);
	void createSize2ColocationsGraph();
public:
	void filterByDistance(float threshold) override;
	void filterByPrevalence(float prevalence) override;
	void getMaximalCliques() override;
	void loadData(DataFeed* data, size_t size, unsigned int types) override;

	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> getInsTable()
	{
		return insTable;
	}

	CPUMiningAlgorithmSeq();
	~CPUMiningAlgorithmSeq();
};

