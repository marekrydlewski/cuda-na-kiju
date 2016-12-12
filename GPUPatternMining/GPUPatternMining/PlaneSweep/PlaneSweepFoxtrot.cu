#pragma once

#include <device_functions.hpp>

#include "thrust/reduce.h"

#include "../HashMap/gpuhashmapper.h"

#include "../MiningCommon.h"


namespace PlaneSweep
{
	namespace Foxtrot
	{
		template<class T>
		__device__ void intraWarpReduce(volatile T *data)
		{
			volatile unsigned int *a = data;
			unsigned int temp;
			int threadInWarp = threadIdx.x;//&0x1f;

			if ((threadIdx.x & 0x01) == 0x01) a[threadInWarp] += a[threadInWarp - 1];
			if ((threadIdx.x & 0x03) == 0x03) a[threadInWarp] += a[threadInWarp - 2];
			if ((threadIdx.x & 0x07) == 0x07) a[threadInWarp] += a[threadInWarp - 4];
			if ((threadIdx.x & 0x0f) == 0x0f) a[threadInWarp] += a[threadInWarp - 8];
			if ((threadIdx.x & 0x1f) == 0x1f) a[threadInWarp] += a[threadInWarp - 16];
		}

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

			// btid
			int blockThreadId = threadIdx.x;
			//gid
			int globalId = computeLinearAddressFrom2D();
			// wid
			int warpId = globalId / 32; 
			// bwid
			int blockWarpId = blockThreadId / 32;
			// wtid
			int warpThreadId = threadIdx.x % 32;
			

			__shared__ volatile bool * flags;
			__shared__ volatile UInt * found;

			if (threadIdx.x == 0)
			{
				flags = static_cast<bool*>(malloc((blockDim.x / 32) * sizeof(bool)));
				found = static_cast<UInt*>(malloc(blockDim.x * uintSize));
			}

			__syncthreads(); // to remove

			found[blockThreadId] = 0;

			//uint start= wid        * ((inSize-1 ) / warpCount ) + max(0, - warpCount  + wid          + (inSize - 1) % warpCount ) + 1;
			int start = warpId * ((count - 1) / warpsCount) + max(0, - warpsCount + warpId + (count  - 1) % warpsCount) + 1;
			//uint stop=(wid         + 1) * ((inSize-1 ) / warpCount ) + max(0, -warpCount  +  (inSize- 1) % warpCount  + wid          + 1);
			int stop = (warpId + 1) * ((count - 1) / warpsCount) + max(0, -warpsCount +  (count - 1) % warpsCount + warpId + 1);

			if (warpId < warpsCount)
			{		
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

							if ((distance(px, py, lx, ly) <= radiusSquared))
							{
								if (ids[i] != ids[localId])
									found[blockThreadId] += 1;
							}
						}

						if (flags[blockWarpId])
						{
							break;
						}
					}
				}

				intraWarpReduce(found);

				if (globalId == warpId * 32)
				{
					resultNeighboursCount[warpId] = found[blockWarpId * 32 + 31];
				}
			}

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
