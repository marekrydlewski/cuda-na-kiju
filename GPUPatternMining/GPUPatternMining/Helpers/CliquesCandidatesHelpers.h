#pragma once
#include <vector>
#include <memory>
#include <thrust/device_vector.h>

namespace Entities
{

	typedef std::vector<unsigned short> CliqueCandidate;
	typedef std::vector<CliqueCandidate> CliquesCandidates;
	// ----------------------------------------------------------------------------------------------

	typedef thrust::device_vector<unsigned short> UShortThrustVector;
	typedef std::shared_ptr<UShortThrustVector> UShortThrustVectorPtr;

	typedef std::vector<UShortThrustVectorPtr> VectorOfUShortThrustVectors;
	typedef std::shared_ptr<VectorOfUShortThrustVectors> VectorOfUShortThrustVectorsPtr;

	typedef thrust::device_vector<thrust::device_ptr<const unsigned short>> CliquesData;
	typedef std::shared_ptr<CliquesData> CliquesDataPtr;
	// ----------------------------------------------------------------------------------------------


	struct GpuCliques
	{
		VectorOfUShortThrustVectorsPtr thrustVectorsOfCliques;
		CliquesDataPtr cliquesData;
		unsigned int currentCliquesSize;
		unsigned int candidatesCount;
	};

	GpuCliques moveCliquesCandidatesToGpu(CliquesCandidates& candidates);

}