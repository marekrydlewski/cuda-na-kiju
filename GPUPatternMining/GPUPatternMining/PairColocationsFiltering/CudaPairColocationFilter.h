#pragma once

#include <thrust/device_vector.h>

#include "../HashMap/gpuhashmapper.h"
//-------------------------------------------------------------------------------

typedef thrust::device_vector<unsigned int> ThrustUIntVector;
typedef GPUHashMapper<unsigned int, unsigned int*, GPUUIntKeyProcessor> UIntTableGpuHashMap;
typedef std::shared_ptr<UIntTableGpuHashMap> UIntTableGpuHashMapPtr;
typedef GPUHashMapper<unsigned int, unsigned int, GPUUIntKeyProcessor> UIntGpuHashMap;
typedef std::shared_ptr<UIntGpuHashMap> UIntGpuHashMapPtr;


class CudaPairColocationFilter
{
public:

	CudaPairColocationFilter(UIntTableGpuHashMapPtr* neighboursMap);

	virtual ~CudaPairColocationFilter();

private:

	UIntTableGpuHashMapPtr* neighboursMap;
};
//-------------------------------------------------------------------------------
