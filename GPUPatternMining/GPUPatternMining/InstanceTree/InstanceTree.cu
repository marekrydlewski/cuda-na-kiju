#include  "InstanceTree.h"
#include "InstanceTreeHelpers.h"
#include <thrust/execution_policy.h>


namespace InstanceTree
{


	typedef thrust::device_vector<unsigned int> UIntThrustVector;
	typedef std::shared_ptr<UIntThrustVector> UIntThrustVectorPtr;

	typedef thrust::device_vector<FeatureInstance> FeatureInstanceThrustVector;
	typedef std::shared_ptr<FeatureInstanceThrustVector> FeatureInstanceThrustVectorPtr;
	


	InstanceTree::InstanceTree(
		PlaneSweepTableInstanceResultPtr planeSweepResult
		, IntanceTablesMapCreator::ITMPackPtr instanceTablePack
		, InstanceTypedNeighboursMapCreator::ITNMPackPtr typedInstanceNeighboursPack
	)
		: planeSweepResult(planeSweepResult)
		, instanceTablePack(instanceTablePack)
		, typedInstanceNeighboursPack(typedInstanceNeighboursPack)
	{
		
	}

	InstanceTreeResultPtr InstanceTree::getInstancesResult(Entities::GpuCliques& cliquesCandidates)
	{
		if (cliquesCandidates.cliquesData->empty())
			return nullptr;

		const unsigned int currentCliquesSize = cliquesCandidates.currentCliquesSize;

		std::vector<FeatureInstanceThrustVectorPtr> instancesElementsInLevelVectors;
		thrust::device_vector<thrust::device_ptr<FeatureInstance>> instancesElementsInLevel;


		thrust::device_vector<thrust::device_ptr<bool>> masksOnLevels;

		thrust::device_vector<thrust::device_ptr<unsigned int>> groupNumbersOnLevels;

		//TODO check if is needed
		thrust::device_vector<thrust::device_ptr<unsigned int>> itemNumbersOnLevels;

		// build first tree level
		thrust::device_vector<unsigned int> pairCounts(cliquesCandidates.candidatesCount);
		
		dim3 insertGrid;
		findSmallest2D(cliquesCandidates.candidatesCount, 256, insertGrid.x, insertGrid.y);

		thrust::device_vector<unsigned int> newEntriesCounts(cliquesCandidates.candidatesCount);

		InstanceTreeHelpers::fillFirstPairCountFromMap <<< insertGrid, 256 >>>(
			instanceTablePack->map->getBean()
			, cliquesCandidates.cliquesData->data().get()
			, cliquesCandidates.candidatesCount
			, newEntriesCounts.data()
		);
		
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		std::vector<InstanceTreeHelpers::ForGroupsResultPtr> levelResults;
		
		auto firstTwoLevelsFg = InstanceTreeHelpers::forGroups(newEntriesCounts);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		// level 0
		levelResults.push_back(firstTwoLevelsFg);
		instancesElementsInLevelVectors.push_back(std::make_shared<FeatureInstanceThrustVector>(firstTwoLevelsFg->threadCount));
		instancesElementsInLevel.push_back(instancesElementsInLevelVectors.back()->data());
		groupNumbersOnLevels.push_back(firstTwoLevelsFg->groupNumbers.data());
		itemNumbersOnLevels.push_back(firstTwoLevelsFg->itemNumbers.data());

		// level 1
		levelResults.push_back(firstTwoLevelsFg);
		instancesElementsInLevelVectors.push_back(std::make_shared<FeatureInstanceThrustVector>(firstTwoLevelsFg->threadCount));
		instancesElementsInLevel.push_back(instancesElementsInLevelVectors.back()->data());
		groupNumbersOnLevels.push_back(firstTwoLevelsFg->groupNumbers.data());
		itemNumbersOnLevels.push_back(firstTwoLevelsFg->itemNumbers.data());

		// fill first two tree levels with FeatureInstance
		dim3 writeFirstTwoLevelsGrid;
		findSmallest2D(firstTwoLevelsFg->threadCount, 256, writeFirstTwoLevelsGrid.x, writeFirstTwoLevelsGrid.y);

		InstanceTreeHelpers::writeFirstTwoLevels <<< writeFirstTwoLevelsGrid, 256 >>> (
			instanceTablePack->map->getBean()
			, cliquesCandidates.cliquesData->data().get()
			, firstTwoLevelsFg->groupNumbers.data()
			, firstTwoLevelsFg->itemNumbers.data()
			, planeSweepResult->pairsA.data()
			, planeSweepResult->pairsB.data()
			, firstTwoLevelsFg->threadCount
			, instancesElementsInLevel[0]
			, instancesElementsInLevel[1]
		);
		
		thrust::device_vector<bool> integrityMask(firstTwoLevelsFg->threadCount, true);

		for (unsigned int currentLevel = 2; currentLevel < currentCliquesSize; ++currentLevel)
		{
			unsigned int previousLevelInstancesCount = instancesElementsInLevelVectors.back()->size();

			dim3 getLevelCounts;
			findSmallest2D(previousLevelInstancesCount, 256, getLevelCounts.x, getLevelCounts.y);

			thrust::device_vector<unsigned int> levelCountsResult(previousLevelInstancesCount);

			InstanceTreeHelpers::fillWithNextLevelCountsFromTypedNeighbour <<< getLevelCounts, 256 >>> (
				typedInstanceNeighboursPack->map->getBean()
				, cliquesCandidates.cliquesData->data().get()
				, thrust::raw_pointer_cast(groupNumbersOnLevels.data())
				, instancesElementsInLevel[currentLevel - 1]
				, previousLevelInstancesCount
				, currentLevel
				, integrityMask.data()
				, levelCountsResult.data()
			);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			auto forGroupsResult = InstanceTreeHelpers::forGroups(levelCountsResult);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			// maintain history
			levelResults.push_back(forGroupsResult);
			instancesElementsInLevelVectors.push_back(std::make_shared<FeatureInstanceThrustVector>(forGroupsResult->threadCount));
			instancesElementsInLevel.push_back(instancesElementsInLevelVectors.back()->data());
			groupNumbersOnLevels.push_back(forGroupsResult->groupNumbers.data());
			itemNumbersOnLevels.push_back(forGroupsResult->itemNumbers.data());

			// zero new elements added - there are no instances of any queried candidates
			if (forGroupsResult->threadCount == 0)
				break;

			dim3 insertLevelInstances;
			findSmallest2D(forGroupsResult->threadCount, 256, insertLevelInstances.x, insertLevelInstances.y);

			InstanceTreeHelpers::fillLevelInstancesFromNeighboursList <<< insertLevelInstances, 256 >>> (
				typedInstanceNeighboursPack->map->getBean()
				, cliquesCandidates.cliquesData->data().get()
				, groupNumbersOnLevels.data().get()
				, forGroupsResult->itemNumbers.data()
				, instancesElementsInLevel[currentLevel - 1]
				, planeSweepResult->pairsB.data()
				, forGroupsResult->threadCount
				, currentLevel
				, instancesElementsInLevel[currentLevel]
			);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			
			dim3 checkCliqueIntegrity = insertLevelInstances;

			unsigned int currentLevelElementsCount = instancesElementsInLevelVectors.back()->size();

			if (integrityMask.size() < currentLevelElementsCount)
				integrityMask = thrust::device_vector<bool>(currentLevelElementsCount);

			InstanceTreeHelpers::markAsPartOfCurrentCliqueInstance <<< checkCliqueIntegrity, 256 >>> (
				typedInstanceNeighboursPack->map->getBean()
				, groupNumbersOnLevels.data().get()
				, instancesElementsInLevel.data().get()
				, instancesElementsInLevel[currentLevel]
				, planeSweepResult->pairsB.data()
				, forGroupsResult->threadCount
				, currentLevel
				, integrityMask.data()
			);
			
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		unsigned int candidatesCount = levelResults.back()->threadCount;

		InstanceTreeResultPtr result = std::make_shared<InstanceTreeResult>();

		if (candidatesCount == 0)
		{
			result->instances = thrust::device_vector<FeatureInstance>(0);
			return result;
		}

		const unsigned int writePosVectorMaxSize = candidatesCount * currentCliquesSize;
		
		thrust::device_vector<unsigned int> writePositions(writePosVectorMaxSize);

		unsigned int consistentCount = InstanceTreeHelpers::fillWritePositionsAndReturnCount(
			integrityMask
			, writePositions
			, candidatesCount
		);

		const unsigned int resultVectorSize = consistentCount * currentCliquesSize;
		result->instances = thrust::device_vector<FeatureInstance>(resultVectorSize);

		dim3 reverseGenerateInstancesDim;
		findSmallest2D(candidatesCount, 256, reverseGenerateInstancesDim.x, reverseGenerateInstancesDim.y);

		InstanceTreeHelpers::reverseGenerateCliquesInstances <<< reverseGenerateInstancesDim, 256 >>>(
			groupNumbersOnLevels.data().get()
			, instancesElementsInLevel.data().get()
			, candidatesCount
			, consistentCount
			, currentCliquesSize
			, integrityMask.data()
			, writePositions.data()
			, result->instances.data()
		);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		return result;
	}
}
// --------------------------------------------------------------------------------------------------
