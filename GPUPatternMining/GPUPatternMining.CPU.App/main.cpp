#include <string>
#include <iostream>

#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "../GPUPatternMining.Contract/Graph.h"
#include "../GPUPatternMining.Contract/Timer.h"
#include "../GPUPatternMining.Contract/Benchmark.h"
#include "../GPUPatternMining.Contract/SimulatedRealDataProvider.h"

#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmParallel.h"

int main()
{
	//input data
	const float threshold = 3;
	const float prevalence = 0.0;

	SimulatedRealDataProvider dataProvider;
	auto data = dataProvider.getTestData(DataSet::Medium);

	//output data
	std::vector<std::vector<unsigned short>> solutionSeq, solutionParallel;

	//////////////////////////////////////////////////
	//benchmarking particular methods

	CPUMiningAlgorithmSeq cpuAlgSeq;
	CPUMiningAlgorithmParallel cpuAlgParallel;
	bmk::benchmark<std::chrono::nanoseconds> bmSeq, bmParallel;

	bmSeq.run_p("load data", 1, [&]() { cpuAlgSeq.loadData(std::get<0>(data), std::get<1>(data), std::get<2>(data)); });
	bmSeq.run_p("filter by distance", 1, [&]() { cpuAlgSeq.filterByDistance(threshold); });
	bmSeq.run_p("filter by prevalence", 1, [&]() { cpuAlgSeq.filterByPrevalence(prevalence); });
	bmSeq.run_p("construct max cliques", 1, [&]() { cpuAlgSeq.constructMaximalCliques(); });
	bmSeq.run_p("filter max cliques", 1, [&]() { solutionSeq = cpuAlgSeq.filterMaximalCliques(prevalence); });

	bmSeq.print("sequential algorithm", std::cout);
	//bmSeq.serialize("CPU seq algorithm", "CPUseq.txt");

	bmParallel.run_p("load data", 1, [&]() { cpuAlgParallel.loadData(std::get<0>(data), std::get<1>(data), std::get<2>(data)); });
	bmParallel.run_p("filter by distance", 1, [&]() { cpuAlgParallel.filterByDistance(threshold); });
	bmParallel.run_p("filter by prevalence", 1, [&]() { cpuAlgParallel.filterByPrevalence(prevalence); });
	bmParallel.run_p("construct max cliques", 1, [&]() { cpuAlgParallel.constructMaximalCliques(); });
	bmParallel.run_p("filter max cliques", 1, [&]() { solutionParallel = cpuAlgParallel.filterMaximalCliques(prevalence); });

	bmParallel.print("parallel algorithm  ", std::cout);
	//bmParallel.serialize("CPU parallel algorithm", "CPUparallel.txt");

	return 0;
}