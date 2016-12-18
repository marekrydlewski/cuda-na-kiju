#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../../GPUPatternMining.Contract/Enity/FeatureInstance.h"

#include "../HashMap/gpuhashmapper.h"
#include "../Common/MiningCommon.h"
#include "../Common/CommonOperations.cuh"

using namespace MiningCommon;
// --------------------------------------------------------------------------------------------------------------------------------------


namespace PlaneSweep
{
	namespace Foxtrot
	{
		

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


		struct FeatureInstanceComparator
		{
			__host__ __device__  bool operator()(const  thrust::tuple<FeatureInstance, FeatureInstance>& o1, const thrust::tuple<FeatureInstance, FeatureInstance>& o2)
			{
				if (o1.get<0>().field < o2.get<0>().field) 
					return true;

				if (o1.get<0>().field == o2.get<0>().field) 
					return o1.get<1>().field < o2.get<1>().field;

				return false;
			}
		};
		// --------------------------------------------------------------------------------------------------------------------------------------


		/*
		@template

			Txy					- type of coordinates values
			Idx					- type of feature instances
			Ptx					- type of container that holds information about neighbours list

		@params

			xCoords				- table with instances x coordinates in ascending order
			yCoords				- table with instances y coordinates
			instances			- table with instances
			count				- count of instances
			distanceTreshold	- maximal distance between two instances for concidering them as neigbours (inclusive <=)
			
		@results
			
			hashMapper			- type pair map to list to those types pairs colocations instances that stores result
		
		@info

			tables are treated as zipped (tables values for specified id belong to one instance)
		*/
		template <class Txy = float, class Idx = FeatureInstance, class Ptx = NeighboursListInfoHolder>
		__host__ void PlaneSweep(
			thrust::device_vector<Txy> xCoords
			, thrust::device_vector<Txy>& yCoords
			, thrust::device_vector<Idx>& instances
			, UInt count
			, Txy distanceTreshold
			, std::shared_ptr<GPUHashMapper<Idx, Ptx, GPUKeyProcessor<Idx>>> resultHashMap
			, thrust::device_vector<Idx>& resultPairsA
			, thrust::device_vector<Idx>& resultPairsB)
		{
			typedef thrust::device_vector<Idx> DeviceVector;

			UInt warpsCount = count;
			thrust::device_vector<UInt> neighboursCount(count);
			dim3 grid;

			findSmallest2D(warpsCount * 32, 256, grid.x, grid.y);

			countNeighbours << < grid, 256 >> > (
				thrust::raw_pointer_cast(xCoords.data())
				, thrust::raw_pointer_cast(yCoords.data())
				, thrust::raw_pointer_cast(instances.data())
				, count
				, distanceTreshold
				, distanceTreshold * distanceTreshold
				, warpsCount
				, thrust::raw_pointer_cast(neighboursCount.data())
			);

			UInt totalPairsCount = neighboursCount[count - 1];
			thrust::exclusive_scan(neighboursCount.begin(), neighboursCount.end(), neighboursCount.begin());
			totalPairsCount += neighboursCount[count - 1];

			resultPairsA = DeviceVector(totalPairsCount);
			resultPairsB = DeviceVector(totalPairsCount);

			findNeighbours << < grid, 256 >> > (
				thrust::raw_pointer_cast(xCoords.data())
				, thrust::raw_pointer_cast(yCoords.data())
				, thrust::raw_pointer_cast(instances.data())
				, count
				, distanceTreshold
				, distanceTreshold * distanceTreshold
				, warpsCount
				, thrust::raw_pointer_cast(neighboursCount.data())
				, thrust::raw_pointer_cast(resultPairsA.data())
				, thrust::raw_pointer_cast(resultPairsB.data())
			);


			MiningCommon::zipSort<FeatureInstance, FeatureInstanceComparator>(
				resultPairsA
				, resultPairsB
			);


			thrust::device_vector<FeatureInstance> uniques(totalPairs);
			thrust::device_vector<UInt> indices(totalPairs);
			thrust::device_vector<UInt> counts(totalPairs);

			UInt entryCount = thrust::reduce_by_key(
				resultPairsA.begin(),
				resultPairsA.end(),
				thrust::make_zip_iterator(
					thrust::make_tuple(
						thrust::counting_iterator<UInt>(0),
						thrust::constant_iterator<UInt>(1)
					)
				),
				uniques.begin(),
				thrust::make_zip_iterator(
					thrust::make_tuple(
						indices.begin(),
						counts.end()
					)
				),
				MiningCommon::InstanceEquality<FeatureInstance>(),
				MiningCommon::FirstIndexAndCount<UInt>()
			) - uniques.begin();

			constexpr float entryCountHashMapMultiplier = 1.5f;

			resultHashMap.reset(new GPUHashMapper<Idx, Ptx, GPUKeyProcessor<Idx>>(
				entryCount * entryCountHashMapMultiplier,
				new  GPUKeyProcessor<unsigned int>())
			);

			dim3 insertGrid;
			findSmallest2D(entryCount, 256, insertGrid.x, insertGrid.y);

			insertIntoHashMap << <grid, 256 >> >(
				resultHashMap->getBean(),
				thrust::raw_pointer_cast(uniques.data())
				, thrust::raw_pointer_cast(indices.data())
				, thrust::raw_pointer_cast(counts.data()),
				entryCount
			);
		}
		// --------------------------------------------------------------------------------------------------------------------------------------
	}
}
