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
	const float threshold = 5;
	const float prevalence = 0.0;

	SimulatedRealDataProvider dataProvider;
	auto data = dataProvider.getTestData();

	//output data
	std::vector<std::vector<unsigned short>> solutionSeq, solutionParallel;

	//////////////////////////////////////////////////
	//benchmarking particular methods

	CPUMiningAlgorithmSeq cpuAlgSeq;
	CPUMiningAlgorithmParallel cpuAlgParallel;
	bmk::benchmark<std::chrono::nanoseconds> bmSeq, bmParallel;

	bmSeq.run("load data", 1, [&]() { cpuAlgSeq.loadData(std::get<0>(data), std::get<1>(data), std::get<2>(data)); });
	bmSeq.run("filter by distance", 1, [&]() { cpuAlgSeq.filterByDistance(threshold); });
	bmSeq.run("filter by prevalence", 1, [&]() { cpuAlgSeq.filterByPrevalence(prevalence); });
	bmSeq.run("construct max cliques", 1, [&]() { cpuAlgSeq.constructMaximalCliques(); });
	bmSeq.run("filter max cliques", 1, [&]() { solutionSeq = cpuAlgSeq.filterMaximalCliques(prevalence); });

	bmSeq.print("sequential algorithm", std::cout);
	//bmSeq.serialize("CPU seq algorithm", "CPUseq.txt");

	bmParallel.run("load data", 1, [&]() { cpuAlgParallel.loadData(std::get<0>(data), std::get<1>(data), std::get<2>(data)); });
	bmParallel.run("filter by distance", 1, [&]() { cpuAlgParallel.filterByDistance(threshold); });
	bmParallel.run("filter by prevalence", 1, [&]() { cpuAlgParallel.filterByPrevalence(prevalence); });
	bmParallel.run("construct max cliques", 1, [&]() { cpuAlgParallel.constructMaximalCliques(); });

	bmParallel.run("filter max cliques", 1, [&]() { solutionParallel = cpuAlgParallel.filterMaximalCliques(prevalence); });

	bmParallel.print("parallel algorithm  ", std::cout);
	//bmParallel.serialize("CPU parallel algorithm", "CPUparallel.txt");

	//////////////////////////////////////////////////
	//benchmark whole algorithm

	//std::cout << tim::detailed_measure<std::chrono::nanoseconds>::execution([&]()
	//{
	//	CPUMiningAlgorithmSeq cpuAlgSeq;
	//	cpuAlgSeq.loadData(data, numberOfInstances, types);
	//	cpuAlgSeq.filterByDistance(threshold);
	//	cpuAlgSeq.filterByPrevalence(prevalence);
	//	cpuAlgSeq.constructMaximalCliques();
	//	solutionSeq = cpuAlgSeq.filterMaximalCliques(prevalence);
	//	std::cout << "CPU Seq: ";
	//}) << std::endl;

	//std::cout << tim::detailed_measure<std::chrono::nanoseconds>::execution([&]()
	//{
	//	CPUMiningAlgorithmParallel cpuAlgParallel;
	//	cpuAlgParallel.loadData(data, numberOfInstances, types);
	//	cpuAlgParallel.filterByDistance(threshold);
	//	cpuAlgParallel.filterByPrevalence(prevalence);
	//	cpuAlgParallel.constructMaximalCliques();
	//	solutionParallel = cpuAlgParallel.filterMaximalCliques(prevalence);
	//	std::cout << "CPU Parallel: ";
	//}) << std::endl;

	return 0;
}