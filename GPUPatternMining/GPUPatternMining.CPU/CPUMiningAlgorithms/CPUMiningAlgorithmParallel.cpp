#include "CPUMiningAlgorithmParallel.h"
#include "../Utilities.h"
#include <ppl.h>
#include <concrtrm.h>

CPUMiningAlgorithmParallel::CPUMiningAlgorithmParallel()
{
}

CPUMiningAlgorithmParallel::~CPUMiningAlgorithmParallel()
{
}

void CPUMiningAlgorithmParallel::loadData(DataFeed * data, size_t size, unsigned int types)
{
	this->typeIncidenceCounter.resize(types, 0);
	this->source.assign(data, data + size);
}

void CPUMiningAlgorithmParallel::filterByDistance(float threshold)
{
	float effectiveThreshold = pow(threshold, 2);

	Concurrency::combinable<std::vector<int>> threadTypeIncidenceCounter;
	//this is going to be parallelized - add parallel_for here
	{
		threadTypeIncidenceCounter.local().resize(typeIncidenceCounter.size());
		//when parallelized add remaining load to last process (occurs when source.size() % GetProcessorCount() != 0)
		unsigned long long int loadPerProcessor = source.size() / concurrency::GetProcessorCount();

		for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
		{
			++threadTypeIncidenceCounter.local()[(*it1).type];

			for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
			{
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

	//results reduction
	typeIncidenceCounter = threadTypeIncidenceCounter.combine([](std::vector<int> left, std::vector<int> right)->std::vector<int> {
		for (int i = 0; i < left.size(); ++i)
		{
			left[i] += right[i];
		}
		return left;
	});

}

void CPUMiningAlgorithmParallel::filterByPrevalence(float prevalence)
{
}

void CPUMiningAlgorithmParallel::constructMaximalCliques()
{
}
