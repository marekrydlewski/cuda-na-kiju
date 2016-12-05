#include "CudaPairColocationFilter.h"



CudaPairColocationFilter::CudaPairColocationFilter(UIntTableGpuHashMapPtr* neighboursMap)
	:neighboursMap(neighboursMap)
{
}


CudaPairColocationFilter::~CudaPairColocationFilter()
{
}
