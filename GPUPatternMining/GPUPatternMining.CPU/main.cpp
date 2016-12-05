#include "../GPUPatternMining.Contract/RandomDataProvider.h"
#include "PairColocationsFiltering/CPUPairColocationsFilter.h"

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

	CPUPairColocationsFilter pairColocationsFilter(data, numberOfInstances, threshold, types);

	return 0;
}