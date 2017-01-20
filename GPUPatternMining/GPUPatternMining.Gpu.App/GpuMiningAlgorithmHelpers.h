#pragma once

#include <memory>
#include <thrust/device_vector.h>
#include "../GPUPatternMining/Entities/FeatureInstance.h"


struct FilterByDistanceGpuData
{
	thrust::device_vector<float> x;
	thrust::device_vector<float> y;
	thrust::device_vector<FeatureInstance> instances;
};

std::shared_ptr<FilterByDistanceGpuData> FilterByDistanceGpuDataPtr;

namespace GpuMiningAlgorithmHelpers
{
	FilterByDistanceGpuDataPtr loadData
}