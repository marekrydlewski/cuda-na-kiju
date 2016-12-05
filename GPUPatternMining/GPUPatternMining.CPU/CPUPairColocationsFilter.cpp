#include "CPUPairColocationsFilter.h"
#include<iostream>
#include<math.h>
#include<vector>

CPUPairColocationsFilter::CPUPairColocationsFilter(DataFeed * data, size_t size, float threshold)
{
	this->effectiveThreshold = pow(threshold, 2);

	std::vector<DataFeed>source(data, data + size);

	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
		{
			if (checkDistance(*it1, *it2))
			{
				//smaller value always first
				auto it1_h = it1;
				auto it2_h = it2;

				if ((*it1_h).type > (*it2_h).type)
					std::swap(it1_h, it2_h);

				if (insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] == nullptr)
				{
					insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] = new std::vector<int>();
				}
				insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId]->push_back((*it2_h).instanceId);
			}
		}
	}
}

void CPUPairColocationsFilter::filterPairColocations(DataFeed * data)
{
	std::cout << "First item: type - " << data[0].type << ", coords: " << data[0].xy.x << " | " << data[0].xy.y << ", instance number - " << data[0].instanceId << ", distance from next one: " << calculateDistance(data[0], data[1]) << std::endl;
}

inline float CPUPairColocationsFilter::calculateDistance(const DataFeed & first, const DataFeed & second)
{
	/// no sqrt coz it is expensive function, there's no need to compute euclides distance, we need only compare values
	return pow(second.xy.x - first.xy.x, 2) + pow(second.xy.y - first.xy.y, 2);
}

inline bool CPUPairColocationsFilter::checkDistance(const DataFeed & first, const DataFeed & second)
{
	return (calculateDistance(first, second) <= effectiveThreshold);
}

DataFeed** CPUPairColocationsFilter::divideAndOrderDataByType(DataFeed * data)
{
	return nullptr;
}
