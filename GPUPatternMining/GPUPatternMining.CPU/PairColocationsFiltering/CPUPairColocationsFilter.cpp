#include "CPUPairColocationsFilter.h"
#include <iostream>
#include <math.h>
#include <vector>

CPUPairColocationsFilter::CPUPairColocationsFilter(DataFeed * data, size_t size, float threshold, unsigned int types)
{
	this->effectiveThreshold = pow(threshold, 2);
	this->typeIncidenceCounter.resize(types + 1, 0);

	std::vector<DataFeed>source(data, data + size);

	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
		{
			++this->typeIncidenceCounter[(*it1).type];

			if ((*it1).type != (*it2).type)
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
						insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] = new std::vector<unsigned int>();
					}
					insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId]->push_back((*it2_h).instanceId);
				}
			}
		}
	}
}

void CPUPairColocationsFilter::filterByPrevalence()
{
	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> typeIncidenceColocations;

	for (auto& a: insTable)
	{
		for (auto& b : a.second)
		{
			auto aType = a.first;
			auto bType = b.first;

			unsigned int aElements = b.second.size();
			unsigned int bElements = 0;

			std::map<unsigned int, bool> inIncidenceColocations;

			for (auto& c : b.second)
			{
				auto aInstance = c.first;
				auto bInstances = c.second;

				for (auto &bInstance : *bInstances)
				{
					if (inIncidenceColocations[bInstance] != true)
					{
						inIncidenceColocations[bInstance] = true;
						++bElements;
					}
				}
			}

			typeIncidenceColocations[std::make_pair(aType, bType)] = std::make_pair(aElements, bElements);
		}
	}
}

inline float CPUPairColocationsFilter::calculateDistance(const DataFeed & first, const DataFeed & second) const
{
	/// no sqrt coz it is expensive function, there's no need to compute euclides distance, we need only compare values
	return pow(second.xy.x - first.xy.x, 2) + pow(second.xy.y - first.xy.y, 2);
}

inline bool CPUPairColocationsFilter::checkDistance(const DataFeed & first, const DataFeed & second) const
{
	return (calculateDistance(first, second) <= effectiveThreshold);
}

DataFeed** CPUPairColocationsFilter::divideAndOrderDataByType(DataFeed * data)
{
	return nullptr;
}
