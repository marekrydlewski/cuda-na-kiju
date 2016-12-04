#include "CPUPairColocationsFilter.h"
#include<iostream>
#include<math.h>

void CPUPairColocationsFilter::FilterPairColocations(DataFeed * data)
{
	std::cout << "First item: type - " << data[0].type << ", coords: " << data[0].xy.x << " | " << data[0].xy.y << ", instance number - " << data[0].instanceId << ", distance from next one: " << CalculateDistance(data[0], data[1]) << std::endl;
}

double CPUPairColocationsFilter::CalculateDistance(DataFeed first, DataFeed second)
{
	return sqrt(pow(second.xy.x - first.xy.x, 2) + pow(second.xy.y - first.xy.y, 2));
}

DataFeed ** CPUPairColocationsFilter::DivideAndOrderDataByType(DataFeed * data)
{
	return nullptr;
}
