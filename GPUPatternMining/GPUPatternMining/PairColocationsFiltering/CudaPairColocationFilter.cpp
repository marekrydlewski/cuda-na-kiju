#include "CudaPairColocationFilter.h"



CudaPairColocationFilter::CudaPairColocationFilter(
	IPairColocationsProviderPtr pairColocationsProvider)
	:pairColocationsProvider(pairColocationsProvider)
{
}


CudaPairColocationFilter::~CudaPairColocationFilter()
{
}
