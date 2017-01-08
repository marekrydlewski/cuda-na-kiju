#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "../GPUPatternMining.Contract/Graph.h"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmParallel.h"

int main()
{
	const unsigned int types = 5;
	const unsigned int rangeY = 100;
	const unsigned int rangeX = 100;
	const unsigned int numberOfInstances = 50;
	const float threshold = 20;
	const float prevalence = 0.2;

	RandomDataProvider rdp;

	rdp.setNumberOfTypes(types);
	rdp.setRange(rangeX, rangeY);

	auto data = rdp.getData(numberOfInstances);

	CPUMiningAlgorithmSeq cpuAlgSeq;
	cpuAlgSeq.loadData(data, numberOfInstances, types);
	cpuAlgSeq.filterByDistance(threshold);
	cpuAlgSeq.filterByPrevalence(prevalence);
	cpuAlgSeq.constructMaximalCliques();
	auto solution = cpuAlgSeq.filterMaximalCliques(prevalence);

	CPUMiningAlgorithmParallel cpuAlgParallel;
	cpuAlgParallel.loadData(data, numberOfInstances, types);
	cpuAlgParallel.filterByDistance(threshold);
	cpuAlgParallel.filterByPrevalence(prevalence);
	cpuAlgParallel.constructMaximalCliques();
	auto solution_parallel = cpuAlgParallel.filterMaximalCliques(prevalence);
	return 0;
}