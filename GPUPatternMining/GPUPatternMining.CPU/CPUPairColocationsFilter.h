#pragma once
#include"../GPUPatternMining.Contract/Enity/DataFeed.h"
#include"../GPUPatternMining.Contract/IPairColocationsFilter.h"

class CPUPairColocationsFilter : public IPairColocationsFilter
{
public:
	void FilterPairColocations(DataFeed* data);
private:
	double CalculateDistance(DataFeed first, DataFeed second);
	DataFeed** DivideAndOrderDataByType(DataFeed* data);
};
