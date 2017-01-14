#pragma once

#include <thrust/device_vector.h>
#include <memory>

#include "../HashMap/gpuhashmapper.h"
#include "../Entities/NeighboursListInfoHolder.h"
#include "../Entities/FeatureInstance.h"
// --------------------------------------------------------------------------


namespace InstanceTypedNeighboursMapCreator
{
	typedef GPUHashMapper<unsigned long long, NeighboursListInfoHolder, GPUULongIntKeyProcessor> TypedNeighboursListMap;
	typedef std::shared_ptr<TypedNeighboursListMap> TypedNeighboursListMapPtr;
	// --------------------------------------------------------------------------

	struct ITNMPack
	{
		TypedNeighboursListMapPtr map;
		thrust::device_vector<unsigned int> begins;
		thrust::device_vector<unsigned int> counts;
		unsigned int count;
	};
	
	typedef std::shared_ptr<ITNMPack> ITNMPackPtr;
	// --------------------------------------------------------------------------


	__host__ __device__
	inline unsigned long long createITNMKey(FeatureInstance instance, unsigned short type)
	{
		return (static_cast<unsigned long long>(instance.field) << 16) | (type);
	}

	ITNMPackPtr createTypedNeighboursListMap(
		thrust::device_vector<FeatureInstance> pairsA
		, thrust::device_vector<FeatureInstance> pairsB
	);

}
