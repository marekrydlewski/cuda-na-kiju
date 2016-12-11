#pragma once

#include "../HashMap/gpuhashmapper.h"

#include "../MiningCommon.h"

namespace PlaneSweep
{
	namespace Foxtrot
	{
		/*
			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			ids					- table with instances ids
			distanceTreshold	- maximal distance between two instances for concidering them as neigbours (inclusive <=)
			count				- count of instances

			neighboursCount		- result table that contains number of neigbours for each feature instance

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy = float, class Idx = unsigned int>
		__global__ void countNeighbours(
			Txy* xCoords
			, Txy* yCoords
			, Idx* types
			, Idx* ids
			, UInt count
			, Txy dis
			, UInt* neighboursCount)
		{
			// not implemented yet
		}
		
		/*
			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			ids					- table with instances ids
			count				- count of instances
			distanceTreshold	- maximal distance between two instances for concidering them as neigbours (inclusive <=)
			hashMapper			- type pair map to list to those types pairs colocations instances that stores result

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy = float, class Idx = unsigned int, class Ptx=unsigned int*>
		__host__ void PlaneSweep(
			Txy* xCoords
			, Txy* yCoords
			, Idx* types
			, Idx* ids
			, UInt count
			, Txy distanceTreshold
			, GPUHashMapper<Idx, Ptx, GPUKeyProcessor<Idx>>& hashMapper)
		{
			UInt* K;
			UInt* V;

			UInt* neighboursCount;


			UInt* I;
			UInt* A;
		}
	}
}
