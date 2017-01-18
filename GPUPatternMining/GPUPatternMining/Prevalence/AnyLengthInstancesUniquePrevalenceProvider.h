#pragma once
#include <vector>
#include "../Entities/TypeCount.h"
#include "../InstanceTree/InstanceTree.h"
// -------------------------------------------------------------------------------------------------

typedef std::vector<unsigned short> CliqueCandidate;
typedef std::vector<CliqueCandidate> CliquesCandidates;
// -------------------------------------------------------------------------------------------------


class AnyLengthInstancesUniquePrevalenceProvider
{
public:

	AnyLengthInstancesUniquePrevalenceProvider(
		TypesCountsPtr typesCounts);

	std::vector<float> getPrevalenceFromCandidatesInstances(
		CliquesCandidates& cliquesCandidates
		, InstanceTree::InstanceTreeResultPtr instanceTreeResult
	);

private:
	TypesCountsPtr typesCounts;
};
// -------------------------------------------------------------------------------------------------
