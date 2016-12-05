#pragma once

#include "../MiningCommon.h"
//-------------------------------------------------------------------------------


class CudaPairColocationFilter
{
public:

	CudaPairColocationFilter(UIntTableGpuHashMapPtr* neighboursMap);

	virtual ~CudaPairColocationFilter();

private:

	UIntTableGpuHashMapPtr* neighboursMap;
};
//-------------------------------------------------------------------------------
