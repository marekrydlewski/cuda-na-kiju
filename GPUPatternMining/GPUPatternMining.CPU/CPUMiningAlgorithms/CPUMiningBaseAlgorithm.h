#pragma once

#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include "../../GPUPatternMining.Contract/IPairColocationsFilter.h"

class CPUMiningBaseAlgorithm
{	
public:

	CPUMiningBaseAlgorithm();
	virtual ~CPUMiningBaseAlgorithm();

	virtual void filterByDistance(float threshold) = 0;
	virtual void filterByPrevalence(float prevalence) = 0;
	virtual void constructMaximalCliques() = 0;
	virtual void loadData(DataFeed* data, size_t size, unsigned short types) = 0;
	virtual std::vector<std::vector<unsigned short>> filterMaximalCliques(float prevalence) = 0;

protected:

	virtual float inline calculateDistance(
		const DataFeed& first,
		const DataFeed& second) const;
	virtual bool inline checkDistance(
		const DataFeed& first,
		const DataFeed& second,
		float effectiveThreshold) const;
	virtual bool inline countPrevalence(
		std::pair<unsigned short, unsigned short> particularInstance,
		std::pair<unsigned short, unsigned short> generalInstance,
		float prevalence) const;
	virtual bool countPrevalence(
		const std::vector<unsigned short>& particularInstances,
		const std::vector<unsigned short>& generalInstances,
		float prevalence) const;
	virtual bool countPrevalenceParallel(
		const std::vector<unsigned short>& particularInstances,
		const std::vector<unsigned short>& generalInstances,
		float prevalence) const;
	virtual std::vector<std::vector<unsigned short>> getAllCliquesSmallerByOne(std::vector<unsigned short>& clique) const;
};

