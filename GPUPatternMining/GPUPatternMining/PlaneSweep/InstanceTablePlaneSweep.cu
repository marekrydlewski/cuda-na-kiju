#include "InstanceTablePlaneSweep.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../Common/MiningCommon.h"

#include "../Entities/InstanceTable.h"
#include "../Common/CommonOperations.h"

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

namespace PlaneSweep
{
	namespace InstanceTable
	{
		__global__ void countNeighbours(
			float* xCoords
			, float* yCoords
			, FeatureInstance* instances
			, int count
			, float radius
			, float radiusSquared
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
			int start = warpId * ((count - 1) / warpsCount) + max(0, -warpsCount + warpId + (count - 1) % warpsCount) + 1;
			//uint stop=(wid         + 1) * ((inSize-1 ) / warpCount ) + max(0, -warpCount  +  (inSize- 1) % warpCount  + wid          + 1);
			int stop = (warpId + 1) * ((count - 1) / warpsCount) + max(0, -warpsCount + warpId + 1 + (count - 1) % warpsCount);

			if (warpId == 63)
			{
				found[blockThreadId] = 0;
			}

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

							if ((MiningCommon::distance(px, py, lx, ly) <= radiusSquared))
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

				MiningCommon::intraWarpReduce(found);

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

		__global__ void findNeighbours(
			float* xCoords
			, float* yCoords
			, FeatureInstance* instances
			, int count
			, float radius
			, float radiusSquared
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

			// const UInt underBuffId = blockWarpId * 64 + warpThreadId;
			// const int aboveBuffId = blockWarpId * 64 + 32 + warpThreadId;

			__shared__ volatile UInt* scanBuf;
			__shared__ volatile bool* flags;
			__shared__ volatile bool* found;
			//volatile __shared__ UInt* buffA;
			//volatile __shared__ UInt* buffB;
			__shared__ UInt* warpBuffPos;


			FeatureInstance temp_a;
			FeatureInstance temp_b;
			// UInt localStart = 0;

			if (threadIdx.x == 0)
			{
				// check Allocating http://www.drdobbs.com/parallel/a-massively-parallel-stack-for-data-allo/240162018?pgno=1
				// measure dynamic allocating in different warps
				scanBuf = static_cast<UInt*>(malloc(blockDim.x * uintSize));
				flags = static_cast<bool*>(malloc((blockDim.x / 32) * sizeof(bool)));
				found = static_cast<bool*>(malloc(blockDim.x * sizeof(bool)));
				//buffA = static_cast<UInt*>(malloc(blockDim.x * sizeof(UInt)));
				//buffB = static_cast<UInt*>(malloc(blockDim.x * sizeof(UInt)));

				warpBuffPos = static_cast<UInt*>(malloc(blockDim.x / 32 * sizeof(UInt)));
			}

			__syncthreads();

			if (warpThreadId == 0)
				warpBuffPos[warpThreadId] = 0;

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
						found[blockThreadId] = false;
						scanBuf[blockThreadId] = 0;

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
								if ((MiningCommon::distance(px, py, lx, ly) <= radiusSquared))
								{
									found[blockThreadId] = true;

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

									scanBuf[blockThreadId] = 1;
								}
							}
						}

						MiningCommon::intraWarpScan<UInt>(scanBuf);
						__syncthreads();

						//if (warpBuffPos[blockWarpId] + scanBuf)

						if (found[blockThreadId])
						{
							int pos = scanBuf[blockThreadId] + outStart;
							out_a[pos] = temp_a;
							out_b[pos] = temp_b;
						}

						outStart += scanBuf[blockWarpId * 32 + 31];

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
				free(const_cast<bool*>(found));
				//free(const_cast<UInt*>(buffA));
				//free(const_cast<UInt*>(buffB));
			}
		}
		// --------------------------------------------------------------------------------------------------------------------------------------

		typedef thrust::tuple<FeatureInstance, FeatureInstance> FeatureInstanceTuple;

		typedef thrust::device_vector<FeatureInstance>::iterator FeatureDeviceVectorIterator;
		typedef thrust::tuple<FeatureDeviceVectorIterator, FeatureDeviceVectorIterator> FeatureInstanceIteratorTuple;
		typedef thrust::zip_iterator<FeatureInstanceIteratorTuple> FeatureInstanceTupleIterator;
		// --------------------------------------------------------------------------------------------------------------------------------------

		struct FeatureInstanceTupleEquality : public thrust::binary_function<FeatureInstanceTuple, FeatureInstanceTuple, bool>
		{
			__host__ __device__ bool operator()(const FeatureInstanceTuple& lhs, const FeatureInstanceTuple& rhs) const
			{
				return lhs.get<0>().fields.featureId == rhs.get<0>().fields.featureId
					&& lhs.get<1>().fields.featureId == rhs.get<1>().fields.featureId;
			}
		};
		//---------------------------------------------------------------------------------------------

		__global__ void InsertFeatureInstanceTupleIntoHashMap(
			HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean,
			FeatureInstanceTuple* keys,
			unsigned int* deltas,
			unsigned int* counts,
			unsigned int count
		)
		{
			unsigned int tid = computeLinearAddressFrom2D();

			if (tid < count)
			{
				GPUHashMapperProcedures::insertKeyValuePair(
					bean,
					(keys[tid].get<0>().field & 0xFFFF0000) | (keys[tid].get<1>().field >> 16) ,
					Entities::InstanceTable(counts[tid], deltas[tid])
				);
			}
		}
		//---------------------------------------------------------------------------------------------

		__host__ void PlaneSweep(
			thrust::device_vector<float> xCoords
			, thrust::device_vector<float>& yCoords
			, thrust::device_vector<FeatureInstance>& instances
			, UInt count
			, float distanceTreshold
			, PlaneSweepTableInstanceResultPtr result)
		{
			UInt warpsCount = count;
			thrust::device_vector<UInt> neighboursCount(count);
			dim3 grid;

			findSmallest2D(warpsCount * 32, 256, grid.x, grid.y);

			countNeighbours <<< grid, 256 >>> (
				thrust::raw_pointer_cast(xCoords.data())
				, thrust::raw_pointer_cast(yCoords.data())
				, thrust::raw_pointer_cast(instances.data())
				, count
				, distanceTreshold
				, distanceTreshold * distanceTreshold
				, warpsCount
				, thrust::raw_pointer_cast(neighboursCount.data())
				);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			
			UInt totalPairsCount = neighboursCount[count - 1];
			thrust::exclusive_scan(neighboursCount.begin(), neighboursCount.end(), neighboursCount.begin());
			totalPairsCount += neighboursCount[count - 1];

			typedef thrust::device_vector<FeatureInstance> InstancesDeviceVector;

			result->pairsA = InstancesDeviceVector(totalPairsCount);
			result->pairsB = InstancesDeviceVector(totalPairsCount);

			findNeighbours <<< grid, 256 >>> (
				thrust::raw_pointer_cast(xCoords.data())
				, thrust::raw_pointer_cast(yCoords.data())
				, thrust::raw_pointer_cast(instances.data())
				, count
				, distanceTreshold
				, distanceTreshold * distanceTreshold
				, warpsCount
				, thrust::raw_pointer_cast(neighboursCount.data())
				, thrust::raw_pointer_cast(result->pairsA.data())
				, thrust::raw_pointer_cast(result->pairsB.data())
				);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			MiningCommon::zipSort(
				result->pairsA
				, result->pairsB
			);			

			FeatureInstanceTupleIterator zippedBegin = thrust::make_zip_iterator(thrust::make_tuple(
				result->pairsA.begin()
				, result->pairsB.begin()
			));
			
			FeatureInstanceTupleIterator zippedEnd = thrust::make_zip_iterator(thrust::make_tuple(
				result->pairsA.end()
				, result->pairsB.end()
			));
			
			result->uniques = thrust::device_vector<FeatureInstanceTuple>(totalPairsCount);
			result->indices = thrust::device_vector<UInt>(totalPairsCount);
			result->counts = thrust::device_vector<UInt>(totalPairsCount);
			
			UInt entryCount = thrust::reduce_by_key(
				zippedBegin,
				zippedEnd,
				thrust::make_zip_iterator(
					thrust::make_tuple(
						thrust::counting_iterator<UInt>(0),
						thrust::constant_iterator<UInt>(1)
					)
				),
				result->uniques.begin(),
				thrust::make_zip_iterator(
					thrust::make_tuple(
						result->indices.begin(),
						result->counts.begin()
					)
				),
				FeatureInstanceTupleEquality(),
				MiningCommon::FirstIndexAndCount<UInt>()
			).first - result->uniques.begin();
			
			constexpr float entryCountHashMapMultiplier = 1.5f;

			result->instanceTableMap.reset(new GPUHashMapper<UInt, Entities::InstanceTable, GPUKeyProcessor<UInt>>(
				entryCount * entryCountHashMapMultiplier,
				new  GPUKeyProcessor<UInt>())
			);

			dim3 insertGrid;
			findSmallest2D(entryCount, 256, insertGrid.x, insertGrid.y);
			
			InsertFeatureInstanceTupleIntoHashMap <<< insertGrid, 256 >>>(
				result->instanceTableMap->getBean(),
				thrust::raw_pointer_cast(result->uniques.data())
				, thrust::raw_pointer_cast(result->indices.data())
				, thrust::raw_pointer_cast(result->counts.data()),
				entryCount
				);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		// --------------------------------------------------------------------------------------------------------------------------------------

	}
}
