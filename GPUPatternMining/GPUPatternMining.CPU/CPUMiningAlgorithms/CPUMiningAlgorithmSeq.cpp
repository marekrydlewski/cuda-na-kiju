#include "CPUMiningAlgorithmSeq.h"

void CPUMiningAlgorithmSeq::loadData(DataFeed * data, size_t size, unsigned int types)
{
	this->typeIncidenceCounter.resize(types + 1, 0);
	this->source.assign(data, data + size);
}


void CPUMiningAlgorithmSeq::filterByDistance(float threshold)
{
	float effectiveThreshold = pow(threshold, 2);

	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
		{
			++this->typeIncidenceCounter[(*it1).type];

			if ((*it1).type != (*it2).type)
			{
				if (checkDistance(*it1, *it2, effectiveThreshold))
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

void CPUMiningAlgorithmSeq::filterByPrevalence(float prevalence)
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

std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> CPUMiningAlgorithmSeq::countUniqueInstances()
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

CPUMiningAlgorithmSeq::CPUMiningAlgorithmSeq():
	CPUMiningBaseAlgorithm()
{
}


CPUMiningAlgorithmSeq::~CPUMiningAlgorithmSeq()
{
}
