#include "AnyLengthInstancesUniquePrevalenceProvider.h"
#include <thrust/execution_policy.h>
#include <thrust/unique.h>


AnyLengthInstancesUniquePrevalenceProvider::AnyLengthInstancesUniquePrevalenceProvider(
	TypesCountsPtr typesCounts)
	: mapKeyProcessor(GPUKeyProcessor<unsigned int>())
{
	typesCountsMap = getGpuTypesCountsMap(typesCounts, &mapKeyProcessor);
}

thrust::host_vector<float> AnyLengthInstancesUniquePrevalenceProvider::getPrevalenceFromCandidatesInstances(
	Entities::GpuCliques cliquesCandidates
	, InstanceTree::InstanceTreeResultPtr instanceTreeResult
) const
{
	const unsigned int candidatesCount = cliquesCandidates.candidatesCount;
	const unsigned int instancesCount = instanceTreeResult->instancesCliqueId.size();

	if (candidatesCount == 0 || instancesCount == 0)
		return std::vector<float>();
	
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

	thrust::sequence(
		thrust::device
		, idxs.begin()
		, idxs.end()
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	unsigned int levels = cliquesCandidates.currentCliquesSize;

	thrust::device_vector<FeatureInstance> levelTempStorage(instancesCount);

	thrust::device_vector<float> gpuResult(candidatesCount, 0);

	auto cliquesTypesCount = getTypesCountOnGpuForCliquesCandidates(
		cliquesCandidates, typesCountsMap
	);

	MinimalCandidatePrevalenceCounter prevalenceCounter;
	{
		prevalenceCounter.data = instanceTreeResult->instances.data();
		prevalenceCounter.begins = begins.data();
		prevalenceCounter.typeCount = cliquesTypesCount->data();
		prevalenceCounter.counts = instancesCounts.data();
		prevalenceCounter.cliqueIds = cliquesID.data();
		prevalenceCounter.levelUniquesTempStorage = levelTempStorage.data();

		prevalenceCounter.results = gpuResult.data();
		prevalenceCounter.levelsCount = cliquesCandidates.currentCliquesSize;
		prevalenceCounter.instancesCount = instancesCount;
		prevalenceCounter.candidatesCount = cliquesCandidates.candidatesCount;

	}

	thrust::for_each(thrust::device, idxs.begin(), idxs.end(), prevalenceCounter);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	return gpuResult;
}
