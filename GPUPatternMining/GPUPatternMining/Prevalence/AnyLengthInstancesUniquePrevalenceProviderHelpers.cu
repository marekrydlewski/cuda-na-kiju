#include "AnyLengthInstancesUniquePrevalenceProviderHelpers.h"
#include <thrust/execution_policy.h>


TypesCountsMapPtr getGpuTypesCountsMap(
	TypesCountsPtr typesCounts
	, GPUKeyProcessor<unsigned int>* mapKeyProcessor
)
{
	auto map = std::make_shared<TypesCountsMap>(typesCounts->size() * 1.5f, mapKeyProcessor);

	std::vector<unsigned int> values;
	std::vector<unsigned int> keys;

	for (TypeCount& tc : *typesCounts)
	{
		keys.push_back(tc.type);
		values.push_back(tc.count);
	}

	thrust::device_vector<unsigned int> gKeys = keys;
	thrust::device_vector<unsigned int> gValues = values;

	map->insertKeyValuePairs(
		gKeys.data().get()
		, gValues.data().get()
		, typesCounts->size()
	);

	return map;
}
// --------------------------------------------------------------------------------------------------

__global__
void fillTypesCountsForCliqueCandidatesInstances(
	TypesCountsMapBean bean
	, thrust::device_ptr<const unsigned short>* cliquesTypes
	, unsigned int count
	, unsigned int candidatesCount
	, thrust::device_ptr<unsigned int> typesCount
)
{
	unsigned int tid = computeLinearAddressFrom2D();
	
	if (tid < count)
	{
		unsigned int level = tid % candidatesCount;
		unsigned int pos = tid / candidatesCount;

		unsigned int key = cliquesTypes[level][pos];
		unsigned int val;

		GPUHashMapperProcedures::getValue(
			bean
			, key
			, val
		);

		typesCount[tid] = val;
	}
}
// --------------------------------------------------------------------------------------------------

UIntDeviceVectorPtr getTypesCountOnGpuForCliquesCandidates(
	Entities::GpuCliques cliquesCandidates
	, TypesCountsMapPtr typesCountsMap
)
{
	unsigned int threadCount = cliquesCandidates.candidatesCount * cliquesCandidates.currentCliquesSize;

	dim3 insertGrid;
	findSmallest2D(threadCount, 256, insertGrid.x, insertGrid.y);

	UIntDeviceVectorPtr result = std::make_shared<UIntDeviceVector>(threadCount);

	fillTypesCountsForCliqueCandidatesInstances << < insertGrid, 256 >> > (
		typesCountsMap->getBean()
		, cliquesCandidates.cliquesData->data().get()
		, threadCount
		, cliquesCandidates.candidatesCount
		, result->data()
		);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	return result;
}
// --------------------------------------------------------------------------------------------------

__host__ __device__
void MinimalCandidatePrevalenceCounter::operator()(unsigned int idx) const
{
	float currentMinimalPrevalence = 1;

	const unsigned int cliqueId = cliqueIds[idx];

	for (unsigned int currentLevel = 0; currentLevel < levelsCount; ++currentLevel)
	{
		float currentResult = thrust::distance
		(
			levelUniquesTempStorage + begins[idx]
			, thrust::unique_copy
			(
				thrust::device
				, data + (instancesCount * currentLevel) + begins[idx]
				, data + (instancesCount * currentLevel) + begins[idx] + counts[idx]
				, levelUniquesTempStorage + begins[idx]
			)
		) / static_cast<float>(typeCount[candidatesCount * currentLevel + cliqueId]);

		if (currentResult < currentMinimalPrevalence)
			currentMinimalPrevalence = currentResult;
	}

	results[cliqueId] = currentMinimalPrevalence;
}
// --------------------------------------------------------------------------------------------------
