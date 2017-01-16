#include  "InstanceTree.h"
#include "InstanceTreeHelpers.h"
#include <thrust/execution_policy.h>


namespace InstanceTree
{
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

	InstanceTreeResultPtr InstanceTree::getInstancesResult(CliquesCandidates cliquesCandidates)
	{
		if (cliquesCandidates.empty())
			return nullptr;

		const unsigned int currentCliquesSize = cliquesCandidates[0].size();

		InstanceTreeResultPtr result = std::make_shared<InstanceTreeResult>();
		
		// migrate data to GPU
		thrust::device_vector<thrust::device_vector<unsigned short>> cliquesData;
		thrust::device_vector<thrust::device_ptr<const unsigned short>> cliques;
		{
			thrust::host_vector<thrust::device_vector<unsigned short>> hcliquesData;

			for (CliqueCandidate cc : cliquesCandidates)
				hcliquesData.push_back(cc);

			cliquesData = hcliquesData;

			std::vector<thrust::device_ptr<const unsigned short>> hcliques;
			for (thrust::device_vector<unsigned short> tdus : hcliquesData)
				hcliques.push_back(tdus.data());

			cliques = hcliques;
		}

		std::vector<thrust::device_vector<FeatureInstance>> instancesElementsInLevelVectors;
		std::vector<thrust::device_ptr<FeatureInstance>> instancesElementsInLevel;


		thrust::device_vector<thrust::device_ptr<bool>> masksOnLevels;

		thrust::device_vector<thrust::device_ptr<unsigned int>> groupNumbersOnLevels;
		thrust::device_vector<thrust::device_ptr<unsigned int>> itemNumbersOnLevels;
		

		// build first tree level
		thrust::device_vector<unsigned int> pairCounts(cliquesCandidates.size());
		
		dim3 insertGrid;
		findSmallest2D(cliquesCandidates.size(), 256, insertGrid.x, insertGrid.y);

		thrust::device_vector<unsigned int> newEntriesCounts(cliquesCandidates.size());

		InstanceTreeHelpers::fillFirstPairCountFromMap <<< insertGrid, 256 >>>(
			instanceTablePack->map->getBean()
			, thrust::raw_pointer_cast(cliques.data())
			, cliques.size()
			, newEntriesCounts.data()
		);
		
		cudaDeviceSynchronize();

		std::vector<InstanceTreeHelpers::ForGroupsResultPtr> levelResults;

		auto firstTwoLevelsFg = InstanceTreeHelpers::forGroups(newEntriesCounts);
		
		// level 0
		instancesElementsInLevelVectors.push_back(thrust::device_vector<FeatureInstance>(firstTwoLevelsFg->threadCount));
		instancesElementsInLevel.push_back(instancesElementsInLevelVectors.back().data());
		groupNumbersOnLevels.push_back(firstTwoLevelsFg->groupNumbers.data());
		itemNumbersOnLevels.push_back(firstTwoLevelsFg->itemNumbers.data());
		levelResults.push_back(firstTwoLevelsFg);

		// level 1
		instancesElementsInLevelVectors.push_back(thrust::device_vector<FeatureInstance>(firstTwoLevelsFg->threadCount));
		instancesElementsInLevel.push_back(instancesElementsInLevelVectors.back().data());
		groupNumbersOnLevels.push_back(firstTwoLevelsFg->groupNumbers.data());
		itemNumbersOnLevels.push_back(firstTwoLevelsFg->itemNumbers.data());
		levelResults.push_back(firstTwoLevelsFg);

		// fill first two tree levels with FeatureInstance
		dim3 writeFirstTwoLevelsGrid;
		findSmallest2D(firstTwoLevelsFg->threadCount, 256, writeFirstTwoLevelsGrid.x, writeFirstTwoLevelsGrid.y);

		InstanceTreeHelpers::writeFirstTwoLevels <<< writeFirstTwoLevelsGrid, 256 >>> (
			instanceTablePack->map->getBean()
			, thrust::raw_pointer_cast(cliques.data())
			, firstTwoLevelsFg->groupNumbers.data()
			, firstTwoLevelsFg->itemNumbers.data()
			, planeSweepResult->pairsA.data()
			, planeSweepResult->pairsB.data()
			, firstTwoLevelsFg->threadCount
			, instancesElementsInLevel[0]
			, instancesElementsInLevel[1]
			);

		// TODO add generating bigger trees

		for (unsigned int currentLevel = 2; currentLevel < currentCliquesSize; ++currentLevel)
		{
			unsigned int elementsCount = instancesElementsInLevelVectors.back().size();

			dim3 levelDim;
			findSmallest2D(elementsCount, 256, levelDim.x, levelDim.y);

			thrust::device_vector<unsigned int> fixMask(elementsCount);
			thrust::device_vector<unsigned int> result(elementsCount);

			InstanceTreeHelpers::fillWithNextLevelCountsFromTypedNeighbour <<< levelDim, 256 >>> (
				typedInstanceNeighboursPack->map->getBean()
				, thrust::raw_pointer_cast(cliques.data())
				, thrust::raw_pointer_cast(groupNumbersOnLevels.data())
				, instancesElementsInLevel[currentLevel - 1]
				, fixMask.data()
				, elementsCount
				, currentLevel
				, result.data()
			);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			auto forGroupsResult = InstanceTreeHelpers::forGroups(result);


			thrust::exclusive_scan(thrust::device
				, fixMask.begin(), fixMask.begin() + elementsCount
				, fixMask.begin());

			

			thrust::transform(thrust::device
				, result.begin(), result.end(), fixMask.begin(), fixMask.begin(), thrust::plus<unsigned int>());

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		// TODO reverse generate instances
		
		return result;
	}
}
// --------------------------------------------------------------------------------------------------
