#pragma once

#include <memory>

#include <thrust/device_vector.h>
#include <thrust/unique.h>

#include "../HashMap/gpuhashmapper.h"
#include "../Entities/TypeCount.h"
#include "../Helpers/CliquesCandidatesHelpers.h"
#include "../Entities/FeatureInstance.h"
// --------------------------------------------------------------------------------------------------

typedef GPUHashMapper<unsigned int, unsigned int, GPUKeyProcessor<unsigned int>> TypesCountsMap;
typedef HashMapperBean<unsigned int, unsigned int, GPUKeyProcessor<unsigned int>> TypesCountsMapBean;
typedef std::shared_ptr<TypesCountsMap> TypesCountsMapPtr;
// --------------------------------------------------------------------------------------------------

struct TypesCountsMapResult
{
	thrust::device_vector<unsigned int> counts;
	TypesCountsMapPtr map;
};

typedef std::shared_ptr<TypesCountsMapResult> TypesCountsMapResultPtr;

TypesCountsMapResultPtr getGpuTypesCountsMap(
	TypesCountsPtr typesCounts
	, GPUKeyProcessor<unsigned int>* mapKeyProcessor
);
// --------------------------------------------------------------------------------------------------

typedef thrust::device_vector<unsigned int> UIntDeviceVector;
typedef std::shared_ptr<UIntDeviceVector> UIntDeviceVectorPtr;
// --------------------------------------------------------------------------------------------------

UIntDeviceVectorPtr getTypesCountOnGpuForCliquesCandidates(
	Entities::GpuCliques cliquesCandidates
	, TypesCountsMapPtr typesCountsMap
);
// --------------------------------------------------------------------------------------------------

struct MinimalCandidatePrevalenceCounter
{
	thrust::device_ptr<FeatureInstance> data;
	thrust::device_ptr<unsigned int> begins;
	thrust::device_ptr<unsigned int> typeCount;
	thrust::device_ptr<unsigned int> counts;
	thrust::device_ptr<unsigned int> cliqueIds;
	thrust::device_ptr<FeatureInstance> levelUniquesTempStorage;

	thrust::device_ptr<float> results;
	unsigned int levelsCount;
	unsigned int instancesCount;
	unsigned int candidatesCount;

	__host__ __device__
	void operator()(unsigned int idx) const;
};
// -------------------------------------------------------------------------------------------------
