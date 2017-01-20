#include "GpuMiningAlgorithm.h"

#include <algorithm>

#include "../GPUPatternMining/PlaneSweep/PlaneSweepTableInstanceResult.h"
#include "../GPUPatternMining/PlaneSweep/InstanceTablePlaneSweep.h"
#include "../GPUPatternMining/Prevalence/PrevalentTypedPairProvider.h"

// ------------------------------------------------------------------------------------------------


void GpuMiningAlgorithm::loadData(DataFeed * data, size_t size, unsigned short types)
{
	typeIncidenceCounter.resize(types, 0);
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
		++typeIncidenceCounter[df.type].count;
}

void GpuMiningAlgorithm::filterPrevalentTypedConnections(float minimalPrevalence)
{
	Prevalence::UniqueFilter::PrevalentTypedPairProvider bppc(
		typeIncidenceCounter, itmPack);

	prevalentTypesConnections = bppc.getPrevalentPairConnections(
		minimalPrevalence
		, planeSweepResult
	);
}