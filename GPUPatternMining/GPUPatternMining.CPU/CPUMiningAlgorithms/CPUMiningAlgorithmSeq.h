#pragma once
#include "CPUMiningBaseAlgorithm.h"

#include "../../GPUPatternMining.Contract/Graph.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/SubcliquesContainer.h"
#include "../../GPUPatternMining.Contract/CliquesContainer.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include "../../GPUPatternMining.Contract/Hashers.h"

#include <unordered_map>
#include <vector>

class CPUMiningAlgorithmSeq :
	public CPUMiningBaseAlgorithm
{
public:

	CPUMiningAlgorithmSeq();
	virtual ~CPUMiningAlgorithmSeq();

	void loadData(
		DataFeed* data,
		size_t size,
		unsigned short types) override;

	void filterByDistance(float threshold) override;
	void filterByPrevalence(float prevalence) override;
	void constructMaximalCliques() override;

	std::vector<std::vector<unsigned short>> filterMaximalCliques(float prevalence);
	std::unordered_map<unsigned short, std::unordered_map<unsigned short, std::unordered_map<unsigned short, std::vector<unsigned short>*>>> getInsTable()
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
	std::vector<int> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::unordered_map<unsigned short,
		std::unordered_map<unsigned short,
			std::unordered_map<unsigned short,
				std::vector<unsigned short>*>>> insTable;

	Graph size2ColocationsGraph;
	SubcliquesContainer prevalentCliquesContainer;
	CliquesContainer lapsedCliquesContainer;
	/// Cm
	std::vector<std::vector<unsigned short>> maximalCliques;

	void createSize2ColocationsGraph();

	std::unordered_map<std::pair <unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>, pair_hash> countUniqueInstances();

	bool filterNodeCandidate(
		unsigned short type,
		unsigned short instanceId,
		std::vector<CinsNode*> const & ancestors);

	bool isCliquePrevalent(
		std::vector<unsigned short>& clique,
		float prevalence);

	std::vector<std::vector<ColocationElem>> constructCondensedTree(const std::vector<unsigned short>& Cm);

	std::vector<std::vector<unsigned short>> getPrevalentMaxCliques(
		std::vector<unsigned short>& clique,
		float prevalence,
		std::vector<std::vector<std::vector<unsigned short>>>& cliquesToProcess);

};

