#pragma once

#include "..\..\GPUPatternMining.Contract\IPairColocationsFilter.h"
#include "..\..\GPUPatternMining.Contract\IPairColocationsProvider.h"
//-------------------------------------------------------------------------------

class CudaPairColocationFilter : public IPairColocationsFilter
{
public:

	CudaPairColocationFilter(
		IPairColocationsProviderPtr pairColocationsProvider);

	virtual ~CudaPairColocationFilter();

	void filterPairColocations() override
	{
		
	}

private:

	IPairColocationsProviderPtr pairColocationsProvider;
};
//-------------------------------------------------------------------------------
