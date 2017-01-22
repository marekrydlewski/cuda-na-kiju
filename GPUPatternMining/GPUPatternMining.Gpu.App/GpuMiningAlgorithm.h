#pragma once

#include "..\GPUPatternMining.Contract/Enity/DataFeed.h"
#include "../GPUPatternMining/PlaneSweep/PlaneSweepTableInstanceResult.h"
#include "../GPUPatternMining/InstanceTree/IntanceTablesMapCreator.h"
#include "../GPUPatternMining/InstanceTree/InstanceTypedNeighboursMapCreator.h"
#include "../GPUPatternMining/Entities/TypeCount.h"
#include "../GPUPatternMining.Contract/Graph.h"
#include "../GPUPatternMining/Prevalence/AnyLengthInstancesUniquePrevalenceProvider.h"
#include "../GPUPatternMining.Contract/CliquesContainer.h"

// ------------------------------------------------------------------------------------------------


class GpuMiningAlgorithm
{
public:
	void loadData(DataFeed * data, size_t size, unsigned short types);

	void filterByDistancePrepareData();

	void filterByDistance(float threshold);

	void filterPrevalentTypedConnectionsPrepareData();

	void filterPrevalentTypedConnections(float minimalPrevalence);

	void constructMaximalCliquesPrepareData();

	void constructMaximalCliques();

	void filterCandidatesByPrevalencePrepareData();

	void filterCandidatesByPrevalence(float minimalPrevalence);

private:

	// data
	std::vector<DataFeed> source;
	/// typeIncidenceCounter - count from 1
	TypesCountsPtr typeIncidenceCounter;

	// helpers stuctures
	struct FilterByDistanceGpuData
	{
		thrust::device_vector<float> x;
		thrust::device_vector<float> y;
		thrust::device_vector<FeatureInstance> instances;
	};

	// steps results
	PlaneSweepTableInstanceResultPtr planeSweepResult;
	std::shared_ptr<FilterByDistanceGpuData> filterByDistanceGpuData;
	IntanceTablesMapCreator::ITMPackPtr itmPack;
	thrust::host_vector<FeatureTypePair> prevalentTypesConnections;
	Graph graphForKerbosh;
	std::map<unsigned int, std::vector<std::vector<unsigned short>>> candidates;

	InstanceTypedNeighboursMapCreator::ITNMPackPtr ITNMPack;
	std::shared_ptr<InstanceTree::InstanceTree> instanceTree;
	SubcliquesContainer prevalentCliques;
	SubcliquesContainer pendingCliques;
	std::shared_ptr<AnyLengthInstancesUniquePrevalenceProvider> anyLengthPrevalenceProvider;
};