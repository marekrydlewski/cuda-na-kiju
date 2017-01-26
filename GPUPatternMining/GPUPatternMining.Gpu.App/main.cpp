#include <chrono>
#include "../GPUPatternMining.Contract/Benchmark.h"
#include "../GPUPatternMining.Contract/SimulatedRealDataProvider.h"
#include "GpuMiningAlgorithm.h"

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		printf("Run with params [file_name] [distance] [prevalence] [ordNumber]\n");
		return 0;
	}

	std::string fileName = argv[1];
	float distance = std::stof(argv[2]);
	float prevalence = std::stof(argv[3]);
	std::string ordNumber = argv[4];

	SimulatedRealDataProvider dataProvider;
	auto data = dataProvider.getTestData(fileName);

	bmk::benchmark<std::chrono::milliseconds> bmkGpu;

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
		alg.filterByDistance(distance);
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

	bmkGpu.run("filter candidates by prevalence (prepare data)", 1, [&]()
	{
		alg.filterCandidatesByPrevalencePrepareData();
	});

	std::list<std::vector<unsigned short>> solution;

	bmkGpu.run("filter candidates by prevalence", 1, [&]()
	{
		solution = alg.filterCandidatesByPrevalence(prevalence, bmkGpu);
	});

	bmkGpu.print("gpu algorithm", std::cout);

	bmkGpu.serializeCsv((fileName + "Gpu" + ordNumber + ".csv").c_str());

	for (auto& cand : solution)
	{
		for (unsigned short us : cand)
			printf("%hu ", us);

		printf("\n");
	}

	return 0;
}
