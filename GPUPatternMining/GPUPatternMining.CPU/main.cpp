#include"../GPUPatternMining.Contract/RandomDataProvider.h"
#include"CPUPairColocationsFilter.h"

int main()
{
	RandomDataProvider rdp;
	//CPUPairColocationsFilter pairColocationsFilter;

	rdp.setNumberOfTypes(8);
	rdp.setRange(100, 100);

	auto data = rdp.getData(100);

	//pairColocationsFilter.filterPairColocations(data);

	system("pause");

	return 0;
}