#include "CPUPairColocationsFilter.h"
#include<iostream>
#include<math.h>
#include<vector>

CPUPairColocationsFilter::CPUPairColocationsFilter(DataFeed * data, size_t size)
{
	std::vector<DataFeed>source(data, data + size);

	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
		{
			//smaller value always first
			if ((*it1).type > (*it2).type)
				std::swap(it1, it2);

			if (insTable[(*it1).type][(*it2).type][(*it1).instanceId] == nullptr)
			{
				insTable[(*it1).type][(*it2).type][(*it1).instanceId] = new std::vector<int>();
			}
			insTable[(*it1).type][(*it2).type][(*it1).instanceId]->push_back((*it2).instanceId);
		}
	}
}

void CPUPairColocationsFilter::filterPairColocations(DataFeed * data)
{
	std::cout << "First item: type - " << data[0].type << ", coords: " << data[0].xy.x << " | " << data[0].xy.y << ", instance number - " << data[0].instanceId << ", distance from next one: " << calculateDistance(data[0], data[1]) << std::endl;
}

float CPUPairColocationsFilter::calculateDistance(DataFeed first, DataFeed second)
{
	return sqrt(pow(second.xy.x - first.xy.x, 2) + pow(second.xy.y - first.xy.y, 2));
}

DataFeed** CPUPairColocationsFilter::divideAndOrderDataByType(DataFeed * data)
{
	return nullptr;
}
