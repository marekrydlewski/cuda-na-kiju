#pragma once
#include "CPUMiningBaseAlgorithm.h"
#include "../../GPUPatternMining.Contract/Graph.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include <map>
#include <vector>
#include <ppl.h>
#include <concurrent_unordered_map.h>
#include <concurrent_vector.h>


class CPUMiningAlgorithmParallel :
	public CPUMiningBaseAlgorithm
{
public:
	CPUMiningAlgorithmParallel();
	virtual ~CPUMiningAlgorithmParallel();

	void loadData(
		DataFeed* data,
		size_t size,
		unsigned short types) override;

	void filterByDistance(float threshold) override;
	void filterByPrevalence(float prevalence) override;
	void constructMaximalCliques() override;
	std::vector<std::vector<unsigned short>> filterMaximalCliques(float prevalence);

	std::map<unsigned short, std::map<unsigned short, std::map<unsigned short, std::vector<unsigned short>*>>> getInsTable()
	{
		return insTable;
	}

	std::vector<std::vector<unsigned short>> getMaximalCliques()
	{
		return maximalCliques;
	}

private:

	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	concurrency::concurrent_vector<unsigned short> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<unsigned short, 
		std::map<unsigned short,
			std::map<unsigned short,
				std::vector<unsigned short>*>>> insTable;
	/// Cm
	std::vector<std::vector<unsigned short>> maximalCliques;
	Graph size2ColocationsGraph;

	std::map<std::pair <unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>> countUniqueInstances();

	bool filterNodeCandidate(
		unsigned short type,
		unsigned short instanceId,
		std::vector<CinsNode*> const & ancestors);
	void createSize2ColocationsGraph();
	std::vector<std::vector<ColocationElem>> constructCondensedTree(const std::vector<unsigned short>& Cm);
	bool isCliquePrevalent(std::vector<unsigned short>& clique, float prevalence);
	std::vector<std::vector<unsigned short>> getPrevalentMaxCliques(std::vector<unsigned short>& clique, float prevalence);
	std::vector<unsigned short> inline getWorkloadForInsTable(unsigned int);
};