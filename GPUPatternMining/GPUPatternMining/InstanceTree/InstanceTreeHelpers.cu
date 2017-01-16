#include "InstanceTreeHelpers.h"

namespace InstanceTreeHelpers
{

	__global__
	void fillFirstPairCountFromMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, unsigned int count
		, thrust::device_ptr<unsigned int> result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			unsigned int key = static_cast<unsigned int>(cliquesCandidates[tid][0]) << 16 | cliquesCandidates[tid][1];
			Entities::InstanceTable localRes;
			GPUHashMapperProcedures::getValue(bean, key, localRes);

			__syncthreads();

			result[tid] = localRes.count;
		}
	}
	// ------------------------------------------------------------------------------

	__global__
	void fillWithNextLevelCountsFromTypedNeighbour(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance> previousLevelInstances
		, thrust::device_ptr<unsigned int> fixMask
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<unsigned int> result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			unsigned int cliqueId = groupNumberLevels[currentLevel - 1][tid];

			// this for in real case is never used beacuse this method shold be invoked
			// only for lvl2 (counting from 0)
			for (int level = currentLevel - 2; level != 0; --level)
			{
				cliqueId = groupNumberLevels[level][cliqueId];
			}

			unsigned long long key = InstanceTypedNeighboursMapCreator::createITNMKey(
				previousLevelInstances[tid]
				, cliquesCandidates[cliqueId][currentLevel]
			);
			

			if (!GPUHashMapperProcedures::containsKey(bean, key))
			{
				fixMask[tid] = 1;
				result[tid] = 0;
			}
			else
			{
				NeighboursListInfoHolder nlih;

				GPUHashMapperProcedures::getValue(bean, key, nlih);

				fixMask[tid] = 0;
				result[tid] = nlih.count;
			}
		}
	}
	// ------------------------------------------------------------------------------

	__global__
	void fillLevelInstancesFromNeighboursList(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<unsigned int> itemNumbers
		, thrust::device_ptr<FeatureInstance> previousLevelInstances
		, thrust::device_ptr<FeatureInstance> pairB
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<FeatureInstance> result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			unsigned int currentGroupId = groupNumberLevels[currentLevel][tid];
			unsigned int cliqueId = currentGroupId;

			printf("tid[%u] cliqueId[%u] lvl[%u]\n", tid, cliqueId, currentLevel);

			for (unsigned int level = currentLevel - 1; level != 0; --level)
			{
				cliqueId = groupNumberLevels[level][cliqueId];
				printf("tid[%u] cliqueId[%u] lvl[%u]\n", tid, cliqueId, level);
			}

			unsigned long long key = InstanceTypedNeighboursMapCreator::createITNMKey(
				previousLevelInstances[currentGroupId]
				, cliquesCandidates[cliqueId][currentLevel]
			);

			NeighboursListInfoHolder nlih;
			GPUHashMapperProcedures::getValue(bean, key, nlih);
				
			printf("tid[%u] start[%u] num[%u]\n", 
				tid, nlih.startIdx, 
				*(thrust::raw_pointer_cast(itemNumbers) + tid));

			result[tid] = pairB[nlih.startIdx + itemNumbers[tid]];
		}
	}
	// ------------------------------------------------------------------------------

	__global__ void scatterOnesAndPositions(unsigned int nGroups, unsigned int nThreads, unsigned int* groupSizes, unsigned int* scannedGroupSizes, unsigned int *ones, unsigned int *positions)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid<nGroups)
		{
			if (scannedGroupSizes[tid]<nThreads)
			{
				atomicAdd(ones + scannedGroupSizes[tid], 1);
				if (groupSizes[tid] != 0)
				{
					positions[scannedGroupSizes[tid]] = scannedGroupSizes[tid];
				}
			}
		}
	}
	// -----------------------------------------------------------------------------

	dim3 computeGrid(unsigned int n, unsigned int bs)
	{
		dim3 res;
		findSmallest2D(n, bs, res.x, res.y);
		return res;
	}
	// -----------------------------------------------------------------------------
	
	__global__ void substractTid(unsigned int nThreads, unsigned int *positions)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < nThreads) 
		{
			positions[tid] = tid - positions[tid];
		}
	}
	// -----------------------------------------------------------------------------

	struct decrease : public thrust::unary_function<unsigned int, unsigned int>
	{
		__host__ __device__ unsigned int operator()(const unsigned int &x) const
		{
			return x - 1;
		}
	};
	// -----------------------------------------------------------------------------

	ForGroupsResultPtr forGroups(
		thrust::device_vector<unsigned int>& groupSizes,
		unsigned int bs)
	{
		unsigned int nGroups = groupSizes.size();

		thrust::device_vector<unsigned int> scannedGroupSizes = thrust::device_vector<unsigned int>(nGroups);//temp

		unsigned int nThreads = groupSizes[nGroups - 1];
		thrust::exclusive_scan(groupSizes.begin(), groupSizes.end(), scannedGroupSizes.begin());
		nThreads += scannedGroupSizes[nGroups - 1];

		ForGroupsResultPtr res = std::make_shared<ForGroupsResult>();

		if (nThreads>0)
		{
			res->groupNumbers = thrust::device_vector<unsigned int>(nThreads);
			res->itemNumbers = thrust::device_vector<unsigned int>(nThreads);

			thrust::fill(res->groupNumbers.begin(), res->groupNumbers.end(), 0);
			thrust::fill(res->itemNumbers.begin(), res->itemNumbers.end(), 0);

			scatterOnesAndPositions <<< computeGrid(nGroups, bs), bs >>>(
				nGroups,
				nThreads,
				groupSizes.data().get(),
				scannedGroupSizes.data().get(),
				res->groupNumbers.data().get(),
				res->itemNumbers.data().get());

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			thrust::inclusive_scan(res->groupNumbers.begin(), res->groupNumbers.end(), res->groupNumbers.begin());
			thrust::inclusive_scan(res->itemNumbers.begin(), res->itemNumbers.end(), res->itemNumbers.begin(), thrust::maximum<unsigned int>());

			thrust::transform(res->groupNumbers.begin(), res->groupNumbers.end(), res->groupNumbers.begin(), decrease());

			substractTid <<< computeGrid(nThreads, bs), bs >>>(nThreads, res->itemNumbers.data().get());
		}

		res->threadCount = nThreads;

		return res;
	}
	// -----------------------------------------------------------------------------


	__global__ 
	void writeFirstTwoLevels(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, thrust::device_ptr<unsigned int> groupNumber
		, thrust::device_ptr<unsigned int> itemNumber
		, thrust::device_ptr<FeatureInstance> pairsA
		, thrust::device_ptr<FeatureInstance> pairsB
		, unsigned int count
		, thrust::device_ptr<FeatureInstance> firstLevel
		, thrust::device_ptr<FeatureInstance> secondLevel
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			auto clique = cliquesCandidates[groupNumber[tid]];

			unsigned int key = (static_cast<unsigned int>(clique[0]) << 16) | clique[1];

			Entities::InstanceTable instanceTable;

			GPUHashMapperProcedures::getValue(
				bean
				, key
				, instanceTable);

			firstLevel[tid] = pairsA[instanceTable.startIdx + itemNumber[tid]];
			secondLevel[tid] = pairsB[instanceTable.startIdx + itemNumber[tid]];
		}
	}
}

