#pragma once

#include <thrust/device_vector.h>

#include "../HashMap/gpuhashmapper.h"
//-------------------------------------------------------------------------------

typedef unsigned int UInt;
typedef UInt* UIntPtr;
typedef UInt* UIntTab;

typedef GPUHashMapper<UInt, UIntTab, GPUUIntKeyProcessor> UIntTableGpuHashMap;
typedef std::shared_ptr<UIntTableGpuHashMap> UIntTableGpuHashMapPtr;
typedef GPUHashMapper<UInt, UInt, GPUUIntKeyProcessor> UIntGpuHashMap;
typedef std::shared_ptr<UIntGpuHashMap> UIntGpuHashMapPtr;

constexpr size_t uintSize = sizeof(unsigned int);
constexpr size_t uintPtrSize = sizeof(unsigned int*);

class CudaPairColocationFilter
{
public:

	CudaPairColocationFilter(UIntTableGpuHashMapPtr* neighboursMap);

	virtual ~CudaPairColocationFilter();

private:

	UIntTableGpuHashMapPtr* neighboursMap;
};
//-------------------------------------------------------------------------------
