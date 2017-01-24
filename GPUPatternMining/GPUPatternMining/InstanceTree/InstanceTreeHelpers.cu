#include "InstanceTreeHelpers.h"
#include <thrust/execution_policy.h>


namespace InstanceTreeHelpers
{
	struct ConvertBoolToUInt : public thrust::unary_function<unsigned int, bool>
	{
		__host__ __device__
		unsigned int operator()(bool bl)
		{
			return bl;
		}
	};

	unsigned int fillWritePositionsAndReturnCount(
		thrust::device_vector<bool>& integrityMask
		, thrust::device_vector<unsigned int>& result
		, unsigned int candidatesCount)
	{
		auto f = ConvertBoolToUInt();

		thrust::transform(
			thrust::device
			, integrityMask.begin()
			, integrityMask.begin() + candidatesCount
			, result.begin()
			, f
		);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		unsigned int last = result[candidatesCount - 1];

		thrust::exclusive_scan(
			thrust::device
			, result.begin()
			, result.begin() + candidatesCount
			, result.begin()
		);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		return last + result[candidatesCount - 1];
	}

	__global__
	void fillFirstPairCountFromMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean
		, thrust::device_ptr<const unsigned short>* cliquesCandidates
		, unsigned int count
		, unsigned int* result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			unsigned int key = (static_cast<unsigned int>(cliquesCandidates[tid][0]) << 16) | cliquesCandidates[tid][1];
			Entities::InstanceTable localRes;
			GPUHashMapperProcedures::getValue(bean, key, localRes);

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
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<bool> integrityMask
		, thrust::device_ptr<unsigned int> result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			if (!integrityMask[tid])
			{
				result[tid] = 0;
				return;
			}

			unsigned int cliqueId = groupNumberLevels[currentLevel - 1][tid];

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
				result[tid] = 0;
			}
			else
			{
				NeighboursListInfoHolder nlih;

				GPUHashMapperProcedures::getValue(bean, key, nlih);
				
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

			//printf("tid[%u] lvl[%u]| clique=%u\n", tid, currentLevel, cliqueId);
			for (signed int level = currentLevel - 1; level > 0; --level)
			{
				cliqueId = groupNumberLevels[level][cliqueId];
				//printf("tid[%u] lvl[%i]| clique=%u\n", tid, level, cliqueId);
			}

			unsigned long long key = InstanceTypedNeighboursMapCreator::createITNMKey(
				previousLevelInstances[currentGroupId]
				, cliquesCandidates[cliqueId][currentLevel]
			);

			NeighboursListInfoHolder nlih;
			GPUHashMapperProcedures::getValue(bean, key, nlih);
				
			result[tid] = pairB[nlih.startIdx + itemNumbers[tid]];
		}
	}
	// ------------------------------------------------------------------------------

	__global__
	void reverseGenerateCliquesInstances(
		thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance>* instancesOnLevels
		, unsigned int instancesCount
		, unsigned int consistentCount
		, unsigned int length
		, thrust::device_ptr<bool> integrityMask
		, thrust::device_ptr<unsigned int> writePositions
		, thrust::device_ptr<FeatureInstance> result
		, thrust::device_ptr<unsigned int> instancesCliqueId
	)
	{
		// TODO tune by buffered write

		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < instancesCount)
		{
			if (!integrityMask[tid])
				return;

			int instanceLevelIdx = tid;

			for (int currentPos = length - 1; currentPos >= 0; --currentPos)
			{
				result[currentPos * consistentCount + writePositions[tid]] = instancesOnLevels[currentPos][instanceLevelIdx];

				// first two positions have same position
				if (currentPos > 1)
					instanceLevelIdx = groupNumberLevels[currentPos][instanceLevelIdx];
			}

			instancesCliqueId[writePositions[tid]] = groupNumberLevels[1][instanceLevelIdx];
		}
	}
	// ------------------------------------------------------------------------------

	__global__
	void markAsPartOfCurrentCliqueInstance(
		InstanceTypedNeighboursMapCreator::TypedNeighboursListMapBean bean
		, thrust::device_ptr<unsigned int>* groupNumberLevels
		, thrust::device_ptr<FeatureInstance>* instancesOnLevels
		, thrust::device_ptr<FeatureInstance> currentLevelInstances
		, thrust::device_ptr<FeatureInstance> pairB
		, unsigned int count
		, unsigned int currentLevel
		, thrust::device_ptr<bool> result
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();
		bool isPartOfClique = false;

		if (tid < count)
		{
			FeatureInstance newFeatureInstance = currentLevelInstances[tid];
			unsigned short newFeatureType = newFeatureInstance.fields.featureId;

			unsigned int currentGroupId = groupNumberLevels[currentLevel][tid];

			for (signed int level = currentLevel - 1; level >= 0; --level)
			{
				FeatureInstance currentFi = instancesOnLevels[level][currentGroupId];
				unsigned long long key = InstanceTypedNeighboursMapCreator::createITNMKey(
					currentFi
					, newFeatureType
				);

				isPartOfClique = false;

				if (!GPUHashMapperProcedures::containsKey(bean, key))
				{
					//printf("tid[%u] lvl[%i] | %#8x have no neighbours of type %#8x\n"
					//	, tid, level, currentFi.field, newFeatureType);
					break;
				}

				NeighboursListInfoHolder nlih;

				GPUHashMapperProcedures::getValue(bean, key, nlih);

				// TODO check dynamic parallelism
				for (unsigned int i = nlih.startIdx; i < nlih.startIdx + nlih.count; ++i)
				{
					if (pairB[i] == newFeatureInstance)
					{
						isPartOfClique = true;
						//printf("tid[%u] lvl[%i] %#8x found %#8x\n"
						//	, tid, level, currentFi.field, newFeatureInstance.field);
						break;
					}
				}

				if (!isPartOfClique)
				{
					//printf("tid[%u] lvl[%i] %#8x have no neighbour %#8x\n"
					//	, tid, level, currentFi.field, newFeatureInstance.field);
					break;
				}

				__syncthreads();
				// first two levels have same positions
				if (level > 1)
					currentGroupId = groupNumberLevels[level][currentGroupId];
			}

			result[tid] = isPartOfClique;
		}
	}
	// ------------------------------------------------------------------------------


	__global__ void scatterOnesAndPositions(
		unsigned int nGroups
		, unsigned int nThreads
		, unsigned int* groupSizes
		, unsigned int* scannedGroupSizes
		, unsigned int *groupsNumbers
		, unsigned int *inGroupsPositions)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid<nGroups)
		{
			if (scannedGroupSizes[tid]<nThreads)
			{
				atomicAdd(groupsNumbers + scannedGroupSizes[tid], 1);
				if (groupSizes[tid] != 0)
				{
					inGroupsPositions[scannedGroupSizes[tid]] = scannedGroupSizes[tid];
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

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

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
		, FeatureInstance* pairsA
		, FeatureInstance* pairsB
		, unsigned int count
		, FeatureInstance* firstLevel
		, FeatureInstance* secondLevel
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			const unsigned short* clique = cliquesCandidates[groupNumber[tid]].get();

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

