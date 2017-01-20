#include <chrono>
#include "../GPUPatternMining.Contract/Benchmark.h"
#include "../GPUPatternMining.Contract/SimulatedRealDataProvider.h"
#include "GpuMiningAlgorithm.h"

int main()
{
	//input data
	const float threshold = 5;
	const float prevalence = 0.0f;

	SimulatedRealDataProvider dataProvider;
	auto data = dataProvider.getTestData();

	bmk::benchmark<std::chrono::nanoseconds> bmkGpu;
	
	GpuMiningAlgorithm alg;

	bmkGpu.run("load data", 1, [&]()
	{
		alg.loadData(std::get<0>(data), std::get<1>(data), std::get<2>(data));
	});

	bmkGpu.run("filter by distance (prepare data)", 1, [&]()
	{
		alg.filterByDistancePrepareData();
	});

	bmkGpu.run("filter by distance", 1, [&]()
	{
		alg.filterByDistance(threshold);
	});

	bmkGpu.run("filter prevalent type connections (prepare data)", 1, [&]()
	{
		alg.filterPrevalentTypedConnectionsPrepareData();
	});
	
	bmkGpu.run("filter prevalent type connections", 1, [&]()
	{
		alg.filterPrevalentTypedConnections(prevalence);
	});
	
	bmkGpu.run("construct candidates (prepare data)", 1, [&]()
	{
		alg.constructMaximalCliquesPrepareData();
	});

	bmkGpu.run("construct candidates", 1, [&]()
	{
		alg.constructMaximalCliques();
	});

	bmkGpu.print("gpu algorithm", std::cout);

	return 0;
}
