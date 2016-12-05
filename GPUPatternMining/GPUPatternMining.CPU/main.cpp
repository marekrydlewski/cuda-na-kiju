#include"../GPUPatternMining.Contract/RandomDataProvider.h"
#include"CPUPairColocationsFilter.h"

int main()
{
	RandomDataProvider rdp;

	rdp.setNumberOfTypes(8);
	rdp.setRange(100, 100);

	auto data = rdp.getData(100);

	CPUPairColocationsFilter pairColocationsFilter(data, 100, 10);
	return 0;
}