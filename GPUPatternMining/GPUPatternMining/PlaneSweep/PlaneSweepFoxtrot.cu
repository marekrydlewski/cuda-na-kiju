#pragma once

#include <device_functions.hpp>

#include "thrust/reduce.h"

#include "../HashMap/gpuhashmapper.h"

#include "../MiningCommon.h"


namespace PlaneSweep
{
	namespace Foxtrot
	{
		template<class Tx>
		__device__ float distance(Tx x1, Tx y1, Tx x2, Tx y2)
		{
			return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
		}

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
			, int count
			, Txy radius
			, Txy radiusSquared
			, UInt* resultNeighboursCount
			, int warpsCount)
		{
			// int warpsCount	  = gridDim.x * blockDim.x / 32;
			int globalTid = computeLinearAddressFrom2D(); // blockIdx.x  * blockDim.x + threadIdx.x;
			int globalWarpId = globalTid / 32;
			int warpThreadId = threadIdx.x % 32;
			int blockThreadId = threadIdx.x;
			int blockWarpId = blockThreadId / 32;

			__shared__ volatile bool * flags;
			__shared__ volatile UInt * found;

			if (threadIdx.x == 0)
			{
				flags = static_cast<bool*>(malloc((blockDim.x / 32) * sizeof(bool)));
				found = static_cast<UInt*>(malloc(blockDim.x * uintSize));
			}

			__syncthreads();

			found[blockThreadId] = 0;

			//uint start= wid        * ((inSize-1 ) / warpCount ) + max(0, - warpCount  + wid          + (inSize - 1) % warpCount ) + 1;
			int start = globalWarpId * ((count - 1) / warpsCount) + max(0, - warpsCount + globalWarpId + (count  - 1) % warpsCount) + 1;
			//uint stop=(wid         + 1) * ((inSize-1 ) / warpCount ) + max(0, -warpCount  +  (inSize- 1) % warpCount  + wid          + 1);
			int stop = (globalWarpId + 1) * ((count - 1) / warpsCount) + max(0, -warpsCount +  (count - 1) % warpsCount + globalWarpId + 1);
			
			/*
			int W = count / 32;
			// int startSegment = wid*(W / warpCount)   + max(0, wid          - (warpCount  -  W % warpCount));
			int start = globalWarpId * (W / warpsCount) + max(0, globalWarpId - (warpsCount - (W % warpsCount)));

			// int endSegment=(wid+1)*(W/warpCount)          + max(0, 1 + wid          - (warpCount  - W % warpCount )) -1;
			int stop = (globalWarpId + 1) * (W / warpsCount) + max(0, 1 + globalWarpId + (warpsCount - W % warpsCount)) -1;
			*/

			if (globalWarpId < warpsCount)
			{//First warp				
				found[blockThreadId] = 0;

				for (UInt i = start; i <= stop; i++)
				{
					float px = xCoords[i];
					float py = yCoords[i];

					flags[blockWarpId] = false;

					for (int j = i - 32; j >= -32; j -= 32)
					{
						int localId = warpThreadId + j;
						if (localId >= 0)
						{
							float lx = xCoords[localId];

							if ((px - lx)>radius)
							{
								flags[blockWarpId] = true;
							}

							float ly = yCoords[localId];
							if ((distance(px, py, lx, ly) <= radiusSquared) && (ids[i] != ids[localId]))
							{
								found[blockThreadId] += 1;
							}
						}

						if (flags[blockWarpId])
						{
							break;
						}
					}

				}

				atomicAdd(&resultNeighboursCount[globalWarpId], found[blockThreadId]);
				/*
				intraWarpReduce<thrust::plus<UInt>, 6>(found, thrust::plus<UInt>());

				if (globalTid == globalWarpId * 32)
				{
					resultNeighboursCount[globalWarpId] = found[blockWarpId * 32 + 31];
				}*/
			}


			/*

			for (UInt i = start; i <= end; ++i)
			{
				if (warpThreadId == 0)
					flags[blockWarpId] = false;

				int j = i - 32;

				while (j >= -32)
				{
					int comparedId = warpThreadId + j;

					if (comparedId >= 0)
					{
						if (xCoords[i] - xCoords[comparedId] > distanceTreshold)
							flags[blockWarpId] = true;

						float sqrtRes = sqrt(pow(xCoords[i] - xCoords[comparedId], 2) + pow(yCoords[i] - yCoords[comparedId], 2));

						if (sqrtRes <= distanceTreshold)
							found[blockThreadId] += 1;
					}

					if (flags[blockWarpId])
						break;

					j -= 32;
				}
			}

			atomicAdd(&resultNeighboursCount[globalWarpId], found[blockThreadId]);

			
			*/

			__syncthreads();

			if (threadIdx.x == 0)
			{
				free(const_cast<bool*>(flags));
				free(const_cast<UInt*>(found));
			}

			__syncthreads();
		}
		// --------------------------------------------------------------------------------------------------------------------------------------

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
		// --------------------------------------------------------------------------------------------------------------------------------------
	}
}
