#include "CPUPairColocationsFilter.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

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

//Filters by prevalence, mutates insTable !!!
void CPUPairColocationsFilter::filterByPrevalence(float prevalence)
{
	auto countedInstances = countUniqueInstances();
	//filtering
	for (auto& a : insTable)
	{
		for (auto& b : a.second)
		{
			auto aType = a.first;
			auto bType = b.first;

			bool isPrevalence = countPrevalence(
				countedInstances[std::make_pair(aType, bType)],
				std::make_pair(typeIncidenceCounter[aType], typeIncidenceCounter[bType]), prevalence);

			if (!isPrevalence)
			{
				for (auto& c : b.second)
				{
					delete c.second;
					//clear vectors' memeory firstly
				}
				insTable[aType][bType].clear();
				//clear all keys
			}
		}
	}
}

inline float CPUPairColocationsFilter::calculateDistance(const DataFeed & first, const DataFeed & second) const
{
	// no sqrt coz it is expensive function, there's no need to compute euclides distance, we need only compare values
	return std::pow(second.xy.x - first.xy.x, 2) + std::pow(second.xy.y - first.xy.y, 2);
}

inline bool CPUPairColocationsFilter::checkDistance(const DataFeed & first, const DataFeed & second) const
{
	return (calculateDistance(first, second) <= effectiveThreshold);
}

bool CPUPairColocationsFilter::countPrevalence(const std::pair<unsigned int, unsigned int>& particularInstance, const std::pair<unsigned int, unsigned int>& generalInstance, float prevalence)
{
	float aPrev = particularInstance.first / (float)generalInstance.first;
	float bPrev = particularInstance.first / (float)generalInstance.second;
	return (prevalence < std::min(aPrev, bPrev));
}

std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> CPUPairColocationsFilter::countUniqueInstances()
{
	std::map<std::pair <unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> typeIncidenceColocations;

	//counting types incidence
	for (auto& a : insTable)
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

	return typeIncidenceColocations;
}