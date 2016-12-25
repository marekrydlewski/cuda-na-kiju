#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "../GPUPatternMining.Contract/Graph.h"
#include "../GPUPatternMining.CPU/CPUMiningAlgorithms/CPUMiningAlgorithmSeq.h"

int main()
{
	const unsigned int types = 5;
	const unsigned int rangeY = 100;
	const unsigned int rangeX = 100;
	const unsigned int numberOfInstances = 50;
	const float threshold = 20;
	const float prevalence = 0.20;

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
	//cpuAlgSeq.testFilterMaxCliques();
	return 0;
}