#pragma once

#include <memory>
#include <thrust/device_vector.h>
#include "../Entities/FeatureInstance.h"
#include "../Entities/FeatureTypePair.h"
#include "../HashMap/gpuhashmapper.h"
#include "../Entities/InstanceTable.h"


namespace IntanceTablesMapCreator
{
	typedef GPUHashMapper<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> InstanceTableMap;
	typedef std::shared_ptr<InstanceTableMap> InstanceTableMapPtr;
	typedef thrust::tuple<FeatureInstance, FeatureInstance> FeatureInstanceTuple;

	struct ITMPack
	{
		InstanceTableMapPtr map;
		thrust::device_vector<unsigned int> begins;
		thrust::device_vector<unsigned int> counts;
		thrust::device_vector<FeatureInstanceTuple> uniques;
		unsigned int count;
	};

	typedef std::shared_ptr<ITMPack> ITMPackPtr;

	ITMPackPtr createTypedNeighboursListMap(
		thrust::device_vector<FeatureInstance> pairsA
		, thrust::device_vector<FeatureInstance> pairsB
	);
}
