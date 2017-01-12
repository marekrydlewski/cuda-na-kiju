#pragma once

#include <memory>
#include <thrust/device_vector.h>

#include "../Entities/FeatureInstance.h"
#include "../Entities/InstanceTable.h"
#include "../HashMap/gpuhashmapper.h"
// --------------------------------------------------------------------------------------------------


class PlaneSweepTableInstanceResult
{
public:

	thrust::device_vector<FeatureInstance> pairsA;
	thrust::device_vector<FeatureInstance> pairsB;
	
	/*
		map key is feature types pair combined in 8 bytes
		0x11112222  :  1111 is type of first in pair
					   2222 is type of second in pair
	*/
	std::shared_ptr<GPUHashMapper<unsigned int, Entities::InstanceTable, GPUKeyProcessor<unsigned int>>> instanceTableMap;

	thrust::device_vector<thrust::tuple<FeatureInstance, FeatureInstance>> uniques;
	thrust::device_vector<unsigned int> indices;
	thrust::device_vector<unsigned int> counts;
};
// --------------------------------------------------------------------------------------------------


typedef std::shared_ptr<PlaneSweepTableInstanceResult> PlaneSweepTableInstanceResultPtr;
// --------------------------------------------------------------------------------------------------