#include "GpuMiningAlgorithm.h"

#include <algorithm>

#include "../GPUPatternMining/PlaneSweep/PlaneSweepTableInstanceResult.h"
#include "../GPUPatternMining/PlaneSweep/InstanceTablePlaneSweep.h"
#include "../GPUPatternMining/Prevalence/PrevalentTypedPairProvider.h"
#include "../GPUPatternMining/Prevalence/AnyLengthInstancesUniquePrevalenceProvider.h"
#include <list>
#include <chrono>
#include "../GPUPatternMining.Contract/Benchmark.h"

// ------------------------------------------------------------------------------------------------


void GpuMiningAlgorithm::loadData(DataFeed * data, size_t size, unsigned short types)
{

	typeIncidenceCounter = std::make_shared<TypesCounts>(types, TypeCount(0,0));
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

	printf("distance connections count %u\n", planeSweepResult->pairsA.size());

	filterByDistanceGpuData.reset();
}

void GpuMiningAlgorithm::filterPrevalentTypedConnectionsPrepareData()
{
	itmPack = IntanceTablesMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	// TODO write kernel

	for (unsigned int i = 0; i < typeIncidenceCounter->size(); ++i)
		typeIncidenceCounter->at(i).type = i;

	
	for (DataFeed& df : source)
		typeIncidenceCounter->at(df.type).count += 1;
}

void GpuMiningAlgorithm::filterPrevalentTypedConnections(float minimalPrevalence)
{
	Prevalence::UniqueFilter::PrevalentTypedPairProvider bppc(
		*typeIncidenceCounter, itmPack);

	prevalentTypesConnections = bppc.getPrevalentPairConnections(
		minimalPrevalence
		, planeSweepResult
	);

	printf("prevalent connections count %u\n", prevalentTypesConnections.size());
}

void GpuMiningAlgorithm::constructMaximalCliquesPrepareData()
{
	graphForKerbosh.setSize(typeIncidenceCounter->size());
	unsigned int edgeCount = 0;

	for (FeatureTypePair& ftp : prevalentTypesConnections)
	{
		graphForKerbosh.addEdge(ftp.types.a, ftp.types.b);
		++edgeCount;
	}

	printf("Typed graph now contains %u edges\n", edgeCount);
}

void GpuMiningAlgorithm::constructMaximalCliques()
{
	CliquesContainer pendingCliques;

	unsigned int count = 0;

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

		for (std::vector<unsigned short >& cnd : generatedCliques)
		{
			if (pendingCliques.checkCliqueExistence(cnd))
				continue;

			++count;

			candidates[cnd.size()].push_back(cnd);
			pendingCliques.insertClique(cnd);
		}
	}

	printf("Maximal cliques count %u\n", count);
}

void GpuMiningAlgorithm::filterCandidatesByPrevalencePrepareData()
{
	ITNMPack = InstanceTypedNeighboursMapCreator::createTypedNeighboursListMap(
		planeSweepResult->pairsA
		, planeSweepResult->pairsB
	);

	keyProc = std::make_shared<GPUUIntKeyProcessor>();

	typesCountsMap = getGpuTypesCountsMap(typeIncidenceCounter, keyProc.get());

	anyLengthPrevalenceProvider = std::make_shared<AnyLengthInstancesUniquePrevalenceProvider>(typesCountsMap);

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

std::list<std::vector<unsigned short>> GpuMiningAlgorithm::filterCandidatesByPrevalence(float minimalPrevalence, bmk::benchmark<std::chrono::milliseconds>& ben)
{
	std::list<std::vector<unsigned short>> result;

	for (auto cands = candidates.rbegin(); cands  != candidates.rend(); ++cands)
	{
		Entities::GpuCliques gpuCliques;
		std::vector<std::vector<unsigned short>> toProcess;

		ben.run_cumulative("duplicated candidates removal", 1, [&]()
		{
			{
				// for removing candidates repeating with earlier candidates
				CliquesContainer pendingCliques;

				for (auto cand : cands->second)
				{
					if (prevalentCliques.checkCliqueExistence(cand) || pendingCliques.checkCliqueExistence(cand))
						continue;

					pendingCliques.insertClique(cand);
					toProcess.push_back(cand);
				}
			}
		});

		if (toProcess.empty())
			continue;

		unsigned int currentCliqueSize = toProcess[0].size();

		if (currentCliqueSize < 2)
			continue;

		// = Entities::moveCliquesCandidatesToGpu(toProcess);

		ben.run_cumulative("build instance tree (prepare data)", 1, [&]()
		{
			gpuCliques = Entities::moveCliquesCandidatesToGpu(toProcess);
		});

		InstanceTree::InstanceTreeResultPtr instanceTreeResult;// = instanceTree->getInstancesResult(gpuCliques);

		ben.run_cumulative("build instance tree", 1, [&]()
		{
			instanceTreeResult = instanceTree->getInstancesResult(gpuCliques);
		});

		std::shared_ptr<thrust::device_vector<float>> mask;

		ben.run_cumulative("calculate candidates prevalence", 1, [&]()
		{
			mask = anyLengthPrevalenceProvider->getPrevalenceFromCandidatesInstances(
				gpuCliques
				, instanceTreeResult
			);
		});

		thrust::host_vector<float> hPrevalences = *mask;
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		ben.run_cumulative("duplicated candidates removal", 1, [&]()
		{
			// for removing new repeating candidates
			CliquesContainer pendingCliques;

			for (int i = 0; i < hPrevalences.size(); ++i)
			{
				//for (unsigned short us : toProcess[i])
				//	printf("%hu ", us);
				//printf(" prev%f\n", hPrevalences[i]);

				if (hPrevalences[i] >= minimalPrevalence)
				{
					prevalentCliques.insertClique(toProcess[i]);

					result.push_back(toProcess[i]);
				}
				else if (currentCliqueSize > 2)
				{
					auto smallerCliques = getAllCliquesSmallerByOne(toProcess[i]);

					for (auto cand : smallerCliques)
					{
						if (pendingCliques.checkCliqueExistence(cand) || prevalentCliques.checkCliqueExistence(cand))
							continue;

						candidates[currentCliqueSize - 1].push_back(cand);
						pendingCliques.insertClique(cand);
					}
				}
			}
		});
	}

	return result;
}