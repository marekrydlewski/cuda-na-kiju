#pragma once
#include "CPUMiningBaseAlgorithm.h"
#include "../../GPUPatternMining.Contract/Graph.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include <map>
#include <vector>
#include <set>

class CPUMiningAlgorithmSeq :
	public CPUMiningBaseAlgorithm
{
public:

	CPUMiningAlgorithmSeq();
	virtual ~CPUMiningAlgorithmSeq();

	void loadData(
		DataFeed* data,
		size_t size,
		unsigned int types) override;

	void filterByDistance(float threshold) override;
	void filterByPrevalence(float prevalence) override;
	void constructMaximalCliques() override;

	std::vector<std::vector<unsigned int>> filterMaximalCliques(float prevalence);
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> getInsTable()
	{
		return insTable;
	}

	std::vector<std::vector<unsigned int>> getMaximalCliques()
	{
		return maximalCliques;
	}

private:

	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	std::vector<int> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> insTable;
	/// Cm
	std::vector<std::vector<unsigned int>> maximalCliques;
	Graph size2ColocationsGraph;

	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> countUniqueInstances();
	bool filterNodeCandidate(
		unsigned int type,
		unsigned int instanceId,
		std::vector<CinsNode*> const & ancestors);
	void createSize2ColocationsGraph();
	std::vector<std::vector<ColocationElem>> constructCondensedTree(const std::vector<unsigned int>& Cm);
	bool isCliquePrevalent(std::vector<unsigned int>& clique, float prevalence);
	std::vector<std::vector<unsigned int>> getPrevalentMaxCliques(std::vector<unsigned int> clique, float prevalence);

};

