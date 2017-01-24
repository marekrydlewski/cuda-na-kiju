#pragma once
#include <host_defines.h>

#include "IntanceTablesMapCreator.h"
#include "InstanceTypedNeighboursMapCreator.h"

namespace  InstanceTreeHelpers
{
	unsigned int fillWritePositionsAndReturnCount(
		thrust::device_vector<bool>& integrityMask
		, thrust::device_vector<unsigned int>& result
		, unsigned int candidatesCount);

	__global__
	void fillFirstPairCountFromMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, unsigned int count
		,unsigned int* result
	);
	// -----------------------------------------------------------------------------
	
	__global__
	void fillWithNextLevelCountsFromTypedNeighbour(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance> previousLevelInstances
		, unsigned int count
		, unsigned int currentLevel
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
		, FeatureInstance* pairsA
		, FeatureInstance* pairsB
		, unsigned int count
		, FeatureInstance* firstLevel
		, FeatureInstance* secondLevel
	);
	// -----------------------------------------------------------------------------

	__global__
	void fillWithNextLevelCountsFromTypedNeighbour(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance> previousLevelInstances
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<bool> integrityMask
		, thrust::device_ptr<unsigned int> result
	);
	// -----------------------------------------------------------------------------

	__global__
	void fillLevelInstancesFromNeighboursList(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<unsigned int> itemNumbers
		, thrust::device_ptr<FeatureInstance> previousLevelInstances
		, thrust::device_ptr<FeatureInstance> pairB
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<FeatureInstance> result
	);
	// -----------------------------------------------------------------------------

	__global__
	void reverseGenerateCliquesInstances(
		thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance>* instancesOnLevels
		, unsigned int instancesCount
		, unsigned int consistentCount
		, unsigned int length
		, thrust::device_ptr<bool> integrityMask
		, thrust::device_ptr<unsigned int> writePositions
		, thrust::device_ptr<FeatureInstance> result
		, thrust::device_ptr<unsigned int> instancesCliqueId
	);
	// -----------------------------------------------------------------------------

	__global__
	void markAsPartOfCurrentCliqueInstance(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance>* instancesOnLevels
		, thrust::device_ptr<FeatureInstance> currentLevelInstances
		, thrust::device_ptr<FeatureInstance> pairB
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<bool> result
	);
	// -----------------------------------------------------------------------------
}
