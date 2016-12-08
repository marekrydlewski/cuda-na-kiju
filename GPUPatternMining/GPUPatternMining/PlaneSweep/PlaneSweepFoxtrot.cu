#pragma once

#include "../HashMap/gpuhashmapper.h"

namespace PlaneSweep
{
	namespace Foxtrot
	{
		/*
			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			ids					- table with instances ids
			count				- count of instances
			maxDistance			- maximal distance between two instances for concidering them as neigbours (inclusive <=)
			hashMapper			- type pair map to list to those types pairs colocations instances that stores result

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy, class Idx = unsigned int, class Ptx=unsigned int*>
		__host__ void PlaneSweep(
			Txy* xCoords
			, Txy* yCoords
			, Idx* types
			, Idx* ids
			, unsigned int count
			, float distanceTreshold
			, GPUHashMapper<Idx, Ptx, GPUKeyProcessor<Idx>>& hashMapper)
		{
			// not implemented yet
		}
	}
}
