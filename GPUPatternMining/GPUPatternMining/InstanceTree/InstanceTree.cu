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

		// build first tree level
		thrust::device_vector<unsigned int> pairCounts(cliquesCandidates.size());
		
		dim3 insertGrid;
		findSmallest2D(cliquesCandidates.size(), 256, insertGrid.x, insertGrid.y);

		thrust::device_vector<unsigned int> newEntriesCounts(cliquesCandidates.size());

		InstanceTreeHelpers::fillFirstPairCountFromMap <<< insertGrid, 256 >>>(
			instanceTablePack->map->getBean()
			, thrust::raw_pointer_cast(cliques.data())
			, 3
			, newEntriesCounts.data()
		);

		unsigned int levelSize = thrust::reduce(
			thrust::device
			, newEntriesCounts.begin()
			, newEntriesCounts.end()
		);



		return result;
	}
}
// --------------------------------------------------------------------------------------------------
