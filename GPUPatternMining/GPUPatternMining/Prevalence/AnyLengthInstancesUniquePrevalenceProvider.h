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
		TypesCountsMapResultPtr typesCountsMap);

	std::shared_ptr<thrust::device_vector<float>> getPrevalenceFromCandidatesInstances(
		Entities::GpuCliques cliquesCandidates
		, InstanceTree::InstanceTreeResultPtr instanceTreeResult
	) const;

private:
	
	TypesCountsMapResultPtr typesCountsMap;
};
// -------------------------------------------------------------------------------------------------
