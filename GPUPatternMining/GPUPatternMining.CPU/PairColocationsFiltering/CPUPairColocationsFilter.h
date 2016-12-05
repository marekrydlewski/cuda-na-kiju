#pragma once
#include <vector>
#include <map>

#include"../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include"../../GPUPatternMining.Contract/IPairColocationsFilter.h"

class CPUPairColocationsFilter : public IPairColocationsFilter
{
public:
	float effectiveThreshold;
	/// typeIncidenceCounter - count from 1
	std::vector<int> typeIncidenceCounter;
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> insTable;

	CPUPairColocationsFilter(DataFeed* data, size_t size, float threshold, unsigned int types);
	void filterByPrevalence(float prevalence = 0.5);

private:
	float inline calculateDistance(const DataFeed& first, const DataFeed& second) const;
	bool inline checkDistance(const DataFeed& first, const DataFeed& second) const;
	bool inline countPrevalence(const std::pair<unsigned int, unsigned int>&, const std::pair<unsigned int, unsigned int>&, float prevalence) const;
	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> countUniqueInstances();
};
