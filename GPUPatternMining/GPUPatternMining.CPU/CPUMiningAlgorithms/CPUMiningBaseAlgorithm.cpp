#include "CPUMiningBaseAlgorithm.h"

#include <algorithm>
#include <ppl.h>

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

inline bool CPUMiningBaseAlgorithm::checkDistance(
	const DataFeed & first,
	const DataFeed & second,
	float effectiveThreshold) const
{
	return (calculateDistance(first, second) <= effectiveThreshold);
}

inline bool CPUMiningBaseAlgorithm::countPrevalence(
	std::pair<unsigned short, unsigned short> particularInstance,
	std::pair<unsigned short, unsigned short> generalInstance,
	float prevalence) const
{
	float aPrev = particularInstance.first / (float)generalInstance.first;
	float bPrev = particularInstance.second / (float)generalInstance.second;
	return prevalence < std::min(aPrev, bPrev);
}

bool CPUMiningBaseAlgorithm::countPrevalence(
	const std::vector<unsigned short>& particularInstances,
	const std::vector<unsigned short>& generalInstances,
	float prevalence) const
{
	std::vector<float> tempMins;
	tempMins.resize(particularInstances.size());
	
	std::transform(
		particularInstances.begin(),
		particularInstances.end(),
		generalInstances.begin(),
		tempMins.begin(),
		[](unsigned short a, unsigned short b) { return (a / (float) b); });

	return prevalence < *std::min_element(tempMins.begin(), tempMins.end());
}

bool CPUMiningBaseAlgorithm::countPrevalenceParallel(
	const std::vector<unsigned short>& particularInstances,
	const std::vector<unsigned short>& generalInstances,
	float prevalence) const
{
	std::vector<float> tempMins;
	tempMins.resize(particularInstances.size());

	concurrency::parallel_transform(
		particularInstances.begin(),
		particularInstances.end(),
		generalInstances.begin(),
		tempMins.begin(),
		[](unsigned short a, unsigned short b) { return (a / (float)b); });

	return prevalence < *std::min_element(tempMins.begin(), tempMins.end());
}

std::vector<std::vector<unsigned short>> CPUMiningBaseAlgorithm::getAllCliquesSmallerByOne(std::vector<unsigned short>& clique) const
{
	std::vector<std::vector<unsigned short>> smallCliques;
	for (auto i = 0; i < clique.size(); ++i)
	{
		std::vector<unsigned short> smallClique;
		for (auto j = 0; j < clique.size(); ++j)
		{
			if (j != i) smallClique.push_back(clique[j]);
		}
		smallCliques.push_back(smallClique);
	}
	return smallCliques;
}
