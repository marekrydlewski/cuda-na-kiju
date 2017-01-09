#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "../GPUPatternMining.Contract/Graph.h"
#include "../GPUPatternMining.Contract/Timer.h"

#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmParallel.h"

int main()
{
	//input data
	const unsigned int types = 5;
	const unsigned int rangeY = 300;
	const unsigned int rangeX = 300;
	const unsigned int numberOfInstances = 4000;
	const float threshold = 20;
	const float prevalence = 0.2;

	RandomDataProvider rdp;
	rdp.setNumberOfTypes(types);
	rdp.setRange(rangeX, rangeY);
	auto data = rdp.getData(numberOfInstances);

	//output data
	std::vector<std::vector<unsigned int>> solutionSeq, solutionParallel;

	//benchmarking
	CPUMiningAlgorithmSeq cpuAlgSeq;
	CPUMiningAlgorithmParallel cpuAlgParallel;

	std::cout << tim::measure<std::chrono::nanoseconds>::execution(cpuAlgSeq.loadData, data, numberOfInstances, types) << std::endl;
	std::cout << tim::measure<std::chrono::nanoseconds>::execution(cpuAlgSeq.loadData, data, numberOfInstances, types) << std::endl;
	std::cout << tim::measure<std::chrono::nanoseconds>::execution(cpuAlgSeq.loadData, data, numberOfInstances, types) << std::endl;
	std::cout << tim::measure<std::chrono::nanoseconds>::execution(cpuAlgSeq.loadData, data, numberOfInstances, types) << std::endl;
	std::cout << tim::measure<std::chrono::nanoseconds>::execution(cpuAlgSeq.loadData, data, numberOfInstances, types) << std::endl;


	//std::cout << measure<std::chrono::nanoseconds>::execution([&]()
	//{
	//	CPUMiningAlgorithmSeq cpuAlgSeq;
	//	cpuAlgSeq.loadData(data, numberOfInstances, types);
	//	cpuAlgSeq.filterByDistance(threshold);
	//	cpuAlgSeq.filterByPrevalence(prevalence);
	//	cpuAlgSeq.constructMaximalCliques();
	//	solutionSeq = cpuAlgSeq.filterMaximalCliques(prevalence);
	//	std::cout << "CPU Seq: ";
	//}) << std::endl;

	//std::cout << measure<std::chrono::nanoseconds>::execution([&]()
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