#pragma once

#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include "../../GPUPatternMining.Contract/IPairColocationsFilter.h"

class CPUMiningBaseAlgorithm
{	
protected:
	virtual float inline calculateDistance(
		const DataFeed& first, 
		const DataFeed& second) const;
	virtual bool inline checkDistance(
		const DataFeed& first,
		const DataFeed& second,
		float effectiveThreshold) const;
	virtual bool inline countPrevalence(
		const std::pair<unsigned int, unsigned int>& particularInstance,
		const std::pair<unsigned int, unsigned int>& generalInstance,
		float prevalence) const;
	virtual bool countPrevalence(
		const std::vector<unsigned int> particularInstances,
		const std::vector<unsigned int> generalInstances,
		float prevalence) const;

public:
	virtual void filterByDistance(float threshold) = 0;
	virtual void filterByPrevalence(float prevalence) = 0;
	virtual void constructMaximalCliques() = 0;
	virtual void loadData(DataFeed* data, size_t size, unsigned int types) = 0;
	CPUMiningBaseAlgorithm();
	~CPUMiningBaseAlgorithm();
};

