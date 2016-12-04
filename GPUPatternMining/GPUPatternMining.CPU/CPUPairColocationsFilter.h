#pragma once
#include <vector>
#include <map>

#include"../GPUPatternMining.Contract/Enity/DataFeed.h"
#include"../GPUPatternMining.Contract/IPairColocationsFilter.h"

class CPUPairColocationsFilter : public IPairColocationsFilter
{
public:
	/// InsTable - 2 dimensional hashtable, where frist 2 indexes are types
	/// the value is a map, where key is number of 1st facility's instanceId and value is a vector of 2nd facility's instancesId 
	std::map<int, std::map<int, std::map<int, std::vector<int>>>> insTable;

	CPUPairColocationsFilter(DataFeed* data, size_t size);
	void filterPairColocations(DataFeed* data);
private:
	float calculateDistance(DataFeed first, DataFeed second);
	DataFeed** divideAndOrderDataByType(DataFeed* data);
};
