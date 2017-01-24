#include "AnyLengthInstancesUniquePrevalenceProvider.h"
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <algorithm>


AnyLengthInstancesUniquePrevalenceProvider::AnyLengthInstancesUniquePrevalenceProvider(
	TypesCountsMapResultPtr typesCountsMap)
	: typesCountsMap(typesCountsMap)
{

}

std::shared_ptr<thrust::device_vector<float>> AnyLengthInstancesUniquePrevalenceProvider::getPrevalenceFromCandidatesInstances(
	Entities::GpuCliques cliquesCandidates
	, InstanceTree::InstanceTreeResultPtr instanceTreeResult
) const
{
	const unsigned int candidatesCount = cliquesCandidates.candidatesCount;
	const unsigned int instancesCount = instanceTreeResult->instancesCliqueId.size();

	auto result = std::make_shared<thrust::device_vector<float>>(candidatesCount, 0.f);

	if (candidatesCount == 0 || instancesCount == 0)
		return result;
	
	thrust::device_vector<unsigned int> cliquesID(candidatesCount);
	thrust::device_vector<unsigned int> instancesCounts(candidatesCount);

	// how many candidates "make it" as instances
	unsigned int existingCandidatesCount = thrust::reduce_by_key(
		thrust::device
		, instanceTreeResult->instancesCliqueId.begin()
		, instanceTreeResult->instancesCliqueId.end()
		, thrust::constant_iterator<unsigned int>(1)
		, cliquesID.begin()
		, instancesCounts.begin()
	).first - cliquesID.begin();

	thrust::device_vector<unsigned int> begins(existingCandidatesCount);

	thrust::exclusive_scan(
		thrust::device
		, instancesCounts.begin()
		, instancesCounts.begin() + existingCandidatesCount
		, begins.begin()
	);

	thrust::device_vector<unsigned int> idxs(existingCandidatesCount);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::sequence(
		thrust::device
		, idxs.begin()
		, idxs.end()
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::device_vector<FeatureInstance> levelTempStorage(instancesCount);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	auto cliquesTypesCount = getTypesCountOnGpuForCliquesCandidates(
		cliquesCandidates, typesCountsMap->map
	);

	MinimalCandidatePrevalenceCounter prevalenceCounter;
	{
		prevalenceCounter.data = instanceTreeResult->instances.data();
		prevalenceCounter.begins = begins.data();
		prevalenceCounter.typeCount = cliquesTypesCount->data();
		prevalenceCounter.counts = instancesCounts.data();
		prevalenceCounter.cliqueIds = cliquesID.data();
		prevalenceCounter.levelUniquesTempStorage = levelTempStorage.data();

		prevalenceCounter.results = result->data();
		prevalenceCounter.levelsCount = cliquesCandidates.currentCliquesSize;
		prevalenceCounter.instancesCount = instancesCount;
		prevalenceCounter.candidatesCount = cliquesCandidates.candidatesCount;
	}

	unsigned int countPerIteration = 10;
	unsigned int currentStart = 0;
	unsigned int currentEnd = existingCandidatesCount % countPerIteration;

	currentEnd = std::min(currentEnd + countPerIteration, existingCandidatesCount);

	while (currentEnd <= existingCandidatesCount)
	{
		thrust::for_each(
			thrust::device
			, idxs.begin() + currentStart
			, idxs.begin() + currentEnd
			, prevalenceCounter);
		currentStart += countPerIteration;
		currentEnd += countPerIteration;

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}
	
	return result;
}
