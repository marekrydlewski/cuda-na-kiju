#include "CPUMiningAlgorithmParallel.h"

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
}
