#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "CPUMiningAlgorithmSeq.h"

int main()
{
	const unsigned int types = 5;
	const unsigned int rangeY = 100;
	const unsigned int rangeX = 100;
	const unsigned int numberOfInstances = 50;
	const float threshold = 20;

	RandomDataProvider rdp;

	rdp.setNumberOfTypes(types);
	rdp.setRange(rangeX, rangeY);

	auto data = rdp.getData(numberOfInstances);

	CPUMiningAlgorithmSeq cpuAlgSeq;
	cpuAlgSeq.loadData(data, numberOfInstances, types);
	cpuAlgSeq.filterByDistance(threshold);
	cpuAlgSeq.filterByPrevalence(0.5);

	return 0;
}