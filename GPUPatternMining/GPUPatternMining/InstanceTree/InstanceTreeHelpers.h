#pragma once
#include <host_defines.h>

#include "IntanceTablesMapCreator.h"

namespace  InstanceTreeHelpers
{
	__global__
	void fillFirstPairCountFromMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, unsigned int count
		, thrust::device_ptr<unsigned int> result
	);
	// -----------------------------------------------------------------------------

	struct ForGroupsResult
	{
		unsigned int threadCount;
		thrust::device_vector<unsigned int> groupNumbers;
		thrust::device_vector<unsigned int> itemNumbers;
	};

	typedef std::shared_ptr<ForGroupsResult> ForGroupsResultPtr;
	// -----------------------------------------------------------------------------

	ForGroupsResultPtr forGroups(
		thrust::device_vector<unsigned int>& groupSizes,
		unsigned int bs = 256);
	// -----------------------------------------------------------------------------

	__global__
		void writeFirstTwoLevels(
			HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
			, thrust::device_ptr<const unsigned short>* cliquesCandidates
			, thrust::device_ptr<unsigned int> groupNumber
			, thrust::device_ptr<unsigned int> itemNumber
			, thrust::device_ptr<FeatureInstance> pairsA
			, thrust::device_ptr<FeatureInstance> pairsB
			, unsigned int count
			, thrust::device_ptr<FeatureInstance> firstLevel
			, thrust::device_ptr<FeatureInstance> secondLevel
		);
	// -----------------------------------------------------------------------------
}
