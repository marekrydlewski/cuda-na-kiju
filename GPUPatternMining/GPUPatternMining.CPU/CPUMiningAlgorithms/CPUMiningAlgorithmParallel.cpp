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
	Concurrency::combinable<std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>>> threadInsTable;

	//this is going to be parallelized - add parallel_for here
	{
		threadTypeIncidenceCounter.local().resize(typeIncidenceCounter.size());
		//when parallelized add remaining load to last process (occurs when source.size() % GetProcessorCount() != 0)
		unsigned long long int loadPerProcessor = source.size() / concurrency::GetProcessorCount();

		//second part needs alteration - note that if you add load at last process it will be wrong, think about it
		//0 needs to be changed to parallel_for index
		std::vector<DataFeed>::iterator beginIterator = source.begin() + 0 * loadPerProcessor;

		for (auto it1 = beginIterator; (it1 != source.end()); ++it1)
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

						if (threadInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] == nullptr)
						{
							threadInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] = new std::vector<unsigned int>();
						}
						threadInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId]->push_back((*it2_h).instanceId);
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

	insTable = threadInsTable.combine([](std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> left, std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> right)->std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>*>>> {
		for (auto it = right.begin(); (it != right.end()); ++it)
		{
			//damn, hard.
			//note: add to left (as left is returned)
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
