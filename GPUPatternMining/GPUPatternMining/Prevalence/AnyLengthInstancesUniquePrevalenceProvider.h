#pragma once
#include <vector>
#include "../Entities/TypeCount.h"
#include "../InstanceTree/InstanceTree.h"
#include "AnyLengthInstancesUniquePrevalenceProviderHelpers.h"
// -------------------------------------------------------------------------------------------------

typedef std::vector<unsigned short> CliqueCandidate;
typedef std::vector<CliqueCandidate> CliquesCandidates;
// -------------------------------------------------------------------------------------------------


class AnyLengthInstancesUniquePrevalenceProvider
{
public:

	AnyLengthInstancesUniquePrevalenceProvider(
		TypesCountsPtr typesCounts);

	thrust::host_vector<float> getPrevalenceFromCandidatesInstances(
		Entities::GpuCliques cliquesCandidates
		, InstanceTree::InstanceTreeResultPtr instanceTreeResult
	) const;

private:
	TypesCountsMapPtr typesCountsMap;
	GPUKeyProcessor<unsigned int> mapKeyProcessor;
};
// -------------------------------------------------------------------------------------------------
