#pragma once
#include <memory>
#include "HashMap/gpuhashmapper.h"

#include "..\GPUPatternMining.Contract/Enity/\FeatureInstance.h"

typedef unsigned int UInt;

typedef GPUHashMapper<UInt, UInt*, GPUUIntKeyProcessor> UIntTableGpuHashMap;
typedef std::shared_ptr<UIntTableGpuHashMap> UIntTableGpuHashMapPtr;
typedef GPUHashMapper<UInt, UInt, GPUUIntKeyProcessor> UIntGpuHashMap;
typedef std::shared_ptr<UIntGpuHashMap> UIntGpuHashMapPtr;

constexpr size_t uintSize = sizeof(unsigned int);
constexpr size_t uintPtrSize = sizeof(unsigned int*);

inline bool operator==(const FeatureInstance& a, const FeatureInstance& b)
{
	return a.fields.featureId == b.fields.featureId && a.fields.instanceId == b.fields.instanceId;
}

__device__
inline bool operator!=(const FeatureInstance& a, const FeatureInstance& b)
{
	return a.fields.featureId != b.fields.featureId || a.fields.instanceId != b.fields.instanceId;
}