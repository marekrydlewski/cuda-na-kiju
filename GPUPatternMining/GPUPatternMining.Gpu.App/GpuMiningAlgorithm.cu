#include "GpuMiningAlgorithm.h"

#include <algorithm>

#include "../GPUPatternMining/PlaneSweep/PlaneSweepTableInstanceResult.h"
#include "../GPUPatternMining/PlaneSweep/InstanceTablePlaneSweep.h"
#include "../GPUPatternMining/Prevalence/PrevalentTypedPairProvider.h"
#include "../GPUPatternMining/Prevalence/AnyLengthInstancesUniquePrevalenceProvider.h"

// ------------------------------------------------------------------------------------------------


void GpuMiningAlgorithm::loadData(DataFeed * data, size_t size, unsigned short types)
{

	typeIncidenceCounter = std::make_shared<TypesCounts>(types, 0);
	source.assign(data, data + size);
}

void GpuMiningAlgorithm::filterByDistancePrepareData()
{
	filterByDistanceGpuData = std::make_shared<FilterByDistanceGpuData>();

	std::vector<float> hx(source.size());
	std::transform(source.begin(), source.end(), hx.begin(),
		[](DataFeed& df)
	{
		return df.xy.x;
	});

	filterByDistanceGpuData->x = hx;

	std::vector<float> hy(source.size());
	std::transform(source.begin(), source.end(), hy.begin(),
		[](DataFeed& df)
	{
		return df.xy.y;
	});

	filterByDistanceGpuData->y = hy;;

	std::vector<FeatureInstance> hInstances(source.size());
	std::transform(source.begin(), source.end(), hInstances.begin(),
		[](DataFeed& df)
	{
		FeatureInstance fi;
		fi.fields.featureId = df.type;
		fi.fields.instanceId = df.instanceId;
		return fi;
	});

	filterByDistanceGpuData->instances = hInstances;

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	PlaneSweep::InstanceTable::SortByXAxis(
		filterByDistanceGpuData->x
		, filterByDistanceGpuData->y
		, filterByDistanceGpuData->instances
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}

void GpuMiningAlgorithm::filterByDistance(float threshold)
{
	planeSweepResult = std::make_shared<PlaneSweepTableInstanceResult>();

	PlaneSweep::InstanceTable::PlaneSweep(
		filterByDistanceGpuData->x
		, filterByDistanceGpuData->y
		, filterByDistanceGpuData->instances
		, filterByDistanceGpuData->instances.size()
		, threshold
		, planeSweepResult
	);

	filterByDistanceGpuData.reset();
}

void GpuMiningAlgorithm::filterPrevalentTypedConnectionsPrepareData()
{
	itmPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	// TODO write kernel
	for (DataFeed& df : source)
		++(*typeIncidenceCounter)[df.type].count;
}

void GpuMiningAlgorithm::filterPrevalentTypedConnections(float minimalPrevalence)
{
	Prevalence::UniqueFilter::PrevalentTypedPairProvider bppc(
		*typeIncidenceCounter, itmPack);

	prevalentTypesConnections = bppc.getPrevalentPairConnections(
		minimalPrevalence
		, planeSweepResult
	);
}

void GpuMiningAlgorithm::constructMaximalCliquesPrepareData()
{
	graphForKerbosh.setSize(typeIncidenceCounter->size());
	unsigned int edgeCount = 0;

	for (FeatureTypePair& ftp : prevalentTypesConnections)
	{
		graphForKerbosh.addEdge(ftp.types.a, ftp.types.b);

		printf("%hu %hu\n", ftp.types.a, ftp.types.b);

		++edgeCount;
	}

	printf("Typed graph now contains %u edges\n", edgeCount);
}

void GpuMiningAlgorithm::constructMaximalCliques()
{
	auto degeneracy = graphForKerbosh.getDegeneracy();
	for (unsigned short const vertex : degeneracy.second)
	{
		std::vector<unsigned short> neighboursWithHigherIndices = graphForKerbosh.getVertexNeighboursOfHigherIndex(vertex);
		std::vector<unsigned short> neighboursWithLowerIndices = graphForKerbosh.getVertexNeighboursOfLowerIndex(vertex);
		std::vector<unsigned short> thisVertex = { vertex };

		auto generatedCliques = graphForKerbosh.bkPivot(
			neighboursWithHigherIndices,
			thisVertex,
			neighboursWithLowerIndices);

		for (std::vector<unsigned short > cnd : generatedCliques)
		{
			candidates[cnd.size()].push_back(cnd);
			pendingCliques.insertClique(cnd);
		}

		//candidates.insert(candidates.end(), generatedCliques.begin(), generatedCliques.end());
	}

	//std::sort(candidates.begin(), candidates.end(), [](std::vector<unsigned short>& a, std::vector<unsigned short>& b)
	//{
	//	return b.size() < a.size();
	//});
}

void GpuMiningAlgorithm::filterCandidatesByPrevalencePrepareData()
{
	ITNMPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	anyLengthPrevalenceProvider = std::make_shared<AnyLengthInstancesUniquePrevalenceProvider>(typeIncidenceCounter);

	instanceTree = std::make_shared<InstanceTree::InstanceTree>(
		planeSweepResult
		, itmPack
		, ITNMPack
	);
}


std::vector<std::vector<unsigned short>> getAllCliquesSmallerByOne(std::vector<unsigned short>& clique)
{
	std::vector<std::vector<unsigned short>> smallCliques;
	for (auto i = 0; i < clique.size(); ++i)
	{
		std::vector<unsigned short> smallClique;
		for (auto j = 0; j < clique.size(); ++j)
		{
			if (j != i) smallClique.push_back(clique[j]);
		}
		smallCliques.push_back(smallClique);
	}
	return smallCliques;
}

void GpuMiningAlgorithm::filterCandidatesByPrevalence(float minimalPrevalence)
{
	/*
	printf("candidates\n");
	for (auto g = candidates.rbegin(); g != candidates.rend(); ++g)
	{
		for (auto cand : g->second)
		{
			for (unsigned int t : cand)
				printf("%hu ", t);

			printf(" | ");
		};
		printf("\n");
	}
	printf("candidates end\n");
	*/

	for (auto cands = candidates.rbegin(); cands  != candidates.rend(); ++cands)
	{
		std::vector<std::vector<unsigned short>> toProcess;

		for (auto cand : cands->second)
		{
			if (prevalentCliques.checkCliqueExistence(cand))
				continue;

			toProcess.push_back(cand);
		}

		if (toProcess.empty())
			continue;

		unsigned int currentCliqueSize = toProcess[0].size();

		if (currentCliqueSize < 2)
			continue;

		auto gpuCliques = Entities::moveCliquesCandidatesToGpu(toProcess);
/*
		printf("=======[%u]========\n", currentCliqueSize);
		for (auto cand : toProcess)
		{
			for (unsigned int t : cand)
				printf("%hu ", t);

			printf(" | ");
		};
		*/
		//printf("\n=================\n");

		auto mask = anyLengthPrevalenceProvider->getPrevalenceFromCandidatesInstances(
			gpuCliques
			, instanceTree->getInstancesResult(gpuCliques)
		);

		for (int i = 0; i < mask.size(); ++i)
		{
			printf("candidate-----|");
			for (unsigned short us : toProcess[i])
				printf("%hu ", us);
			printf("|\n");

			if (mask[i] >= minimalPrevalence)
			{
				prevalentCliques.insertClique(toProcess[i]);

				printf("|");
				for (unsigned short us : toProcess[i])
					printf("%hu ", us);
				printf("| exists\n");

			}
			else if (currentCliqueSize > 2)
			{

				auto smallerCliques = getAllCliquesSmallerByOne(toProcess[i]);

				for (auto cand : smallerCliques)
				{
					if (pendingCliques.checkCliqueExistence(cand))
						continue;

					candidates[currentCliqueSize - 1].push_back(cand);
					pendingCliques.insertClique(cand);
				}
			}
		}
		//printf("=================\n");
	}
}