#include "CPUMiningBaseAlgorithm.h"

#include <algorithm>

CPUMiningBaseAlgorithm::CPUMiningBaseAlgorithm()
{
}

CPUMiningBaseAlgorithm::~CPUMiningBaseAlgorithm()
{
}

inline float CPUMiningBaseAlgorithm::calculateDistance(const DataFeed & first, const DataFeed & second) const
{
	// no sqrt coz it is expensive function, there's no need to compute euclides distance, we need only compare values
	return std::pow(second.xy.x - first.xy.x, 2) + std::pow(second.xy.y - first.xy.y, 2);
}

inline bool CPUMiningBaseAlgorithm::checkDistance(const DataFeed & first, const DataFeed & second, float effectiveThreshold) const
{
	return (calculateDistance(first, second) <= effectiveThreshold);
}

inline bool CPUMiningBaseAlgorithm::countPrevalence(const std::pair<unsigned int, unsigned int>& particularInstance, const std::pair<unsigned int, unsigned int>& generalInstance, float prevalence) const
{
	float aPrev = particularInstance.first / (float)generalInstance.first;
	float bPrev = particularInstance.first / (float)generalInstance.second;
	return (prevalence < std::min(aPrev, bPrev));
}
