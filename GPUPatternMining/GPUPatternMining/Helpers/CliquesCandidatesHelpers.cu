#include "CliquesCandidatesHelpers.h"


namespace Entities
{
	GpuCliques moveCliquesCandidatesToGpu(CliquesCandidates& candidates)
	{
		GpuCliques result;

		result.candidatesCount = candidates.size();

		result.thrustVectorsOfCliques = std::make_shared<VectorOfUShortThrustVectors>();

		if (result.candidatesCount == 0)
			return result;

		result.currentCliquesSize = candidates[0].size();

		for (CliqueCandidate cc : candidates)
			result.thrustVectorsOfCliques->push_back(std::make_shared<UShortThrustVector>(cc));

		std::vector<thrust::device_ptr<const unsigned short>> hcliques;
		for (UShortThrustVectorPtr tdus : *result.thrustVectorsOfCliques)
			hcliques.push_back(tdus->data());

		result.cliquesData = std::make_shared<CliquesData>(hcliques);

		return result;
	}
}