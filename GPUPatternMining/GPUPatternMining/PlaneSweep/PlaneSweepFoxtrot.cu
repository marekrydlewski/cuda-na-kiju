#pragma once

#include <device_functions.hpp>
#include <climits>

#include "thrust/reduce.h"

#include "../../GPUPatternMining.Contract/Enity/FeatureInstance.h"

#include "../MiningCommon.h"
#include "../HashMap/gpuhashmapper.h"



namespace PlaneSweep
{
	namespace Foxtrot
	{
		template<class T>
		__device__ void intraWarpReduce(volatile T *data)
		{
			volatile T* a = data;
			int threadInWarp = threadIdx.x;//&0x1f;
			
			/*
				binary sum performed by sequence of odd warp threads
			*/
			if ((threadInWarp & 0x01) == 0x01) a[threadInWarp] += a[threadInWarp - 1];
			if ((threadInWarp & 0x03) == 0x03) a[threadInWarp] += a[threadInWarp - 2];
			if ((threadInWarp & 0x07) == 0x07) a[threadInWarp] += a[threadInWarp - 4];
			if ((threadInWarp & 0x0f) == 0x0f) a[threadInWarp] += a[threadInWarp - 8];
			if ((threadInWarp & 0x1f) == 0x1f) a[threadInWarp] += a[threadInWarp - 16];
		}

		//
		template<class T>
		__device__ void intraWarpScan(volatile T* data)
		{
			volatile T* a = data;
			T temp;
			int threadInWarp = threadIdx.x;//&0x1f;

			//Phase1
			/*
				binary sum performed by sequence of odd warp threads
			*/
			if ((threadInWarp & 0x01) == 0x01) a[threadInWarp] += a[threadInWarp - 1];
			if ((threadInWarp & 0x03) == 0x03) a[threadInWarp] += a[threadInWarp - 2];
			if ((threadInWarp & 0x07) == 0x07) a[threadInWarp] += a[threadInWarp - 4];
			if ((threadInWarp & 0x0f) == 0x0f) a[threadInWarp] += a[threadInWarp - 8];
			//if ((threadIdx.x&0x1f)==0x1f) a[threadInWarp]+=a[threadInWarp-16];

			// TODO check if this is realy important, cause we can just use = 9 lines below
			// zeroing every warp with id = n * 31 where n is any positive total number
			if ((threadInWarp & 0x1f) == 0x1f) a[threadInWarp] = 0;

			//Phase2
			// every warp with id = n * 31 where n is any positive total number do :
			if ((threadIdx.x & 0x1f) == 0x1f)
			{
				temp = a[threadInWarp];
				a[threadInWarp] += a[threadInWarp - 16];
				a[threadInWarp - 16] = temp;
			}

			// TODO check if __syncThread is mandatory here

			if ((threadIdx.x & 0x0f) == 0x0f)
			{
				temp = a[threadInWarp];
				a[threadInWarp] += a[threadInWarp - 8];
				a[threadInWarp - 8] = temp;
			}
			if ((threadIdx.x & 0x07) == 0x07)
			{
				temp = a[threadInWarp];
				a[threadInWarp] += a[threadInWarp - 4];
				a[threadInWarp - 4] = temp;
			}
			if ((threadIdx.x & 0x03) == 0x03)
			{
				temp = a[threadInWarp];
				a[threadInWarp] += a[threadInWarp - 2];
				a[threadInWarp - 2] = temp;
			}
			if ((threadIdx.x & 0x01) == 0x01)
			{
				temp = a[threadInWarp];
				a[threadInWarp] += a[threadInWarp - 1];
				a[threadInWarp - 1] = temp;
			}
		}

		template<class Tx>
		__device__ float distance(Tx x1, Tx y1, Tx x2, Tx y2)
		{
			return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
		}

		/*
		@params
			
			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			instances			- table with feature instances
			count				- count of instances
			radius				- maximal distance between two instances for concidering them as neigbours (inclusive <=) - value for comparision in x-axis
			radiusSquared		- square value of radius - value for comparision cartesian distances between instances
			warpsCount			- more warps means less values to iterate in for loop per warp

		@result

			resultNeighboursCount	- result table that contains number of neigbours for each feature instance

		@info

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy = float>
		__global__ void countNeighbours(
			Txy* xCoords
			, Txy* yCoords
			, FeatureInstance* instances
			, int count
			, Txy radius
			, Txy radiusSquared
			, int warpsCount
			, UInt* resultNeighboursCount)
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

			__syncthreads();

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

							if ((px - lx) > radius)
							{
								flags[blockWarpId] = true;
							}

							float ly = yCoords[localId];

							if ((distance(px, py, lx, ly) <= radiusSquared))
							{
								if (instances[i] != instances[localId])
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

				// warpThreadId == 0?
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
		}
		// --------------------------------------------------------------------------------------------------------------------------------------

		/*
		@params

			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			instances			- table with feature instances
			count				- count of instances
			radius				- maximal distance between two instances for concidering them as neigbours (inclusive <=) - value for comparision in x-axis
			radiusSquared		- square value of radius - value for comparision cartesian distances between instances
			warpsCount			- more warps means less values to iterate in for loop per warp
			outStarts			- result from cuntNeighbours method

		@result

			out_a				- first values of pairs
			out_b				- first values of pairs

		@info

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy = float>
		__global__ void findNeighbours(
			Txy* xCoords
			, Txy* yCoords
			, FeatureInstance* instances
			, int count
			, Txy radius
			, Txy radiusSquared
			, int warpsCount
			, UInt *outStarts
			, FeatureInstance* out_a
			, FeatureInstance* out_b)
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

			const UInt underBuffId = blockWarpId * 64 + warpThreadId;
			const int aboveBuffId = blockWarpId * 64 + 32 + warpThreadId;

			__shared__ volatile UInt* scanBuf;
			__shared__ volatile bool* flags;
			__shared__ volatile UInt* found;
			//volatile __shared__ UInt* buffA;
			//volatile __shared__ UInt* buffB;
			__shared__ UInt* progress;


			FeatureInstance temp_a;
			FeatureInstance temp_b;
			UInt localStart = 0;
	
			if (threadIdx.x == 0)
			{
				// check Allocating http://www.drdobbs.com/parallel/a-massively-parallel-stack-for-data-allo/240162018?pgno=1
				// measure dynamic allocating in different warps
				scanBuf = static_cast<UInt*>(malloc(blockDim.x * uintSize));
				flags = static_cast<bool*>(malloc((blockDim.x / 32) * sizeof(bool)));
				found = static_cast<UInt*>(malloc(blockDim.x * uintSize));
				// buffA = static_cast<UInt*>(malloc(blockDim.x * sizeof(UInt) * 2));
				// buffB = static_cast<UInt*>(malloc(blockDim.x * sizeof(UInt) * 2));

				progress = static_cast<UInt*>(malloc(blockDim.x / 32 * sizeof(UInt)));
			}

			__syncthreads();

			if (warpThreadId == 0)
				progress[warpThreadId] = 0;

			//uint start= wid        * ((inSize-1 ) / warpCount ) + max(0, - warpCount  + wid          + (inSize - 1) % warpCount ) + 1;
			int start = warpId * ((count - 1) / warpsCount) + max(0, -warpsCount + warpId + (count - 1) % warpsCount) + 1;
			//uint stop=(wid         + 1) * ((inSize-1 ) / warpCount ) + max(0, -warpCount  +  (inSize- 1) % warpCount  + wid          + 1);
			int stop = (warpId + 1) * ((count - 1) / warpsCount) + max(0, -warpsCount + (count - 1) % warpsCount + warpId + 1);

			if (warpId < warpsCount)
			{
				UInt outStart = outStarts[warpId];

				for (UInt i = start; i <= stop; i++)
				{
					float px = xCoords[i];
					float py = yCoords[i];

					flags[blockWarpId] = false;

					for (int j = i - 32; j >= -32; j -= 32)
					{
						int localId = warpThreadId + j;
						found[blockThreadId] = 0;

						if (localId >= 0)
						{
							float lx = xCoords[localId];

							if ((px - lx) > radius)
							{
								flags[blockWarpId] = true;
							}

							float ly = yCoords[localId];

							if (instances[i] != instances[localId])
							{
								if ((distance(px, py, lx, ly) <= radiusSquared))
								{
									found[blockThreadId] = 1;
									
									if (instances[i].fields.featureId < instances[localId].fields.featureId
										|| instances[i].fields.instanceId < instances[localId].fields.instanceId)
									{
										temp_a = instances[i];
										temp_b = instances[localId];
									}
									else
									{
										temp_a = instances[localId];
										temp_b = instances[i];
									}

									int pos = atomicInc(&progress[blockWarpId], UINT_MAX) + outStart;

									out_a[pos] = temp_a;
									out_b[pos] = temp_b;
								}
							}
						}

						/*
						scanBuf[blockThreadId] = found[blockThreadId];
						intraWarpScan(scanBuf);

						UInt oldLocalStart = localStart;

						if (found[blockThreadId])
						{
							UInt index = blockWarpId * 64 + (localStart + scanBuf[blockThreadId]) % 64;
							buffA[index] = temp_a.field;
							buffB[index] = temp_b.field;
						
						}

						// (
						//		localstart
						//		+ last value from scanbuff for last thread in warp
						//		+ last value from found for last thread in warp
						//	) mod 64
						localStart = (localStart + scanBuf[blockWarpId * 32 + 31] + found[blockWarpId * 32 + 31]) % 64; 

						if (oldLocalStart < 32 && localStart >= 32)
						{								
							out_a[outStart + warpThreadId].field = buffA[underBuffId];
							out_b[outStart + warpThreadId].field = buffB[underBuffId];
							outStart += 32;
						} 
						else if (localStart < 32)
						{
							out_a[outStart + warpThreadId].field = buffA[aboveBuffId];
							out_b[outStart + warpThreadId].field = buffB[aboveBuffId];
							outStart += 32;
						}
						*/

						if (flags[blockWarpId])
						{
							break;
						}
					}
				}

				/*
				if (localStart < 32 && warpThreadId < localStart)
				{
					out_a[outStart + warpThreadId].field = buffA[underBuffId];
					out_b[outStart + warpThreadId].field = buffB[underBuffId];
				}
				else if (localStart >= 32 && warpThreadId < localStart - 32)
				{
					out_a[outStart + warpThreadId].field = buffA[aboveBuffId];
					out_b[outStart + warpThreadId].field = buffB[aboveBuffId];
				}
				*/
			}

			__syncthreads();

			if (threadIdx.x == 0)
			{
				free(const_cast<UInt*>(scanBuf));
				free(const_cast<bool*>(flags));
				free(const_cast<UInt*>(found));
				//free(const_cast<UInt*>(buffA));
				//free(const_cast<UInt*>(buffB));
			}
		}
		// --------------------------------------------------------------------------------------------------------------------------------------

		/*

		@params

			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			types				- table with instances types
			ids					- table with instances ids
			count				- count of instances
			distanceTreshold	- maximal distance between two instances for concidering them as neigbours (inclusive <=)
			
		@result
			
			hashMapper			- type pair map to list to those types pairs colocations instances that stores result
		
		@info

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
