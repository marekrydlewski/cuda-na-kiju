#include "CPUMiningAlgorithmSeqV2.h"

#include "../../GPUPatternMining.Contract/CinsTree.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <chrono>
#include <map>
#include <unordered_map>

std::vector<std::vector<unsigned short>> CPUMiningAlgorithmSeqV2::filterMaximalCliques(float prevalence)
{
	std::vector<std::vector<unsigned short>> finalMaxCliques;

	std::vector<std::vector<unsigned short>> cliquesToProcess;

	for (auto& cl : maximalCliques)
	{
		cliquesToProcess.push_back(cl);
	}

	std::sort(cliquesToProcess.begin(), cliquesToProcess.end(), [](auto& left, auto& right) {
		return left.size() < right.size();
	});
	
	while (cliquesToProcess.size() != 0)
	{
		auto clique = cliquesToProcess.back();
		cliquesToProcess.pop_back();

		if (clique.size() <= 2)
			finalMaxCliques.push_back(clique);
		else {
			auto maxCliques = getPrevalentMaxCliques(clique, prevalence, cliquesToProcess);
			if (maxCliques.size() != 0)
				insertToFinalWithSubcliquesRemoval(finalMaxCliques, maxCliques);
		}

	}

	return finalMaxCliques;
}

std::vector<std::vector<unsigned short>> CPUMiningAlgorithmSeqV2::getPrevalentMaxCliques(
	std::vector<unsigned short>& clique,
	float prevalence,
	std::vector<std::vector<unsigned short>> & cliquesToProcess)
{
	std::vector<std::vector<unsigned short>> finalMaxCliques;

	if (!prevalentCliquesContainer.checkCliqueExistence(clique) && !lapsedCliquesContainer.checkSubcliqueExistence(clique) )
	{
		if (isCliquePrevalent(clique, prevalence))
		{
			finalMaxCliques.push_back(clique);
			prevalentCliquesContainer.insertClique(clique);
		}
		else
		{
			lapsedCliquesContainer.insertClique(clique);

			if (clique.size() > 2)
			{
				auto smallerCliques = getAllCliquesSmallerByOne(clique);
				if (smallerCliques[0].size() == 2) //no need to construct tree, already checked by filterByPrevalence
				{
					for (auto& smallClique : smallerCliques)
					{
						if (!prevalentCliquesContainer.checkCliqueExistence(smallClique))
						{
							finalMaxCliques.push_back(smallClique);
							prevalentCliquesContainer.insertClique(smallClique);
						}
					}
				}
				else
				{
					for (auto& smallClique : smallerCliques)
					{
						cliquesToProcess.push_back(smallClique);
					}
				}
			}
		}
	}

	return finalMaxCliques;
}

void CPUMiningAlgorithmSeqV2::insertToFinalWithSubcliquesRemoval(std::vector<std::vector<unsigned short>>& setToInsertTo, std::vector<std::vector<unsigned short>> cliquesToBeInserted)
{
	for (auto clique : cliquesToBeInserted)
	{
		CliquesContainer clq;
		clq.insertCliques(setToInsertTo);
		if (clq.checkSubcliqueExistence(clique))
		{
			auto smallerCliques = getAllCliquesSmallerByOne(clique);
			while (smallerCliques.size() != 0)
			{
				auto currCl = smallerCliques.back();
				smallerCliques.pop_back();
				std::vector<std::vector<unsigned short>>::iterator it;
				if ((it = std::find(setToInsertTo.begin(), setToInsertTo.end(), currCl)) != setToInsertTo.end())
				{
					setToInsertTo.erase(it);
				}
				else
				{
					auto evenSmallerCliques = getAllCliquesSmallerByOne(currCl);
					smallerCliques.insert(smallerCliques.end(), evenSmallerCliques.begin(), evenSmallerCliques.end());
				}
			}
		}
		setToInsertTo.push_back(clique);
	}
}

CPUMiningAlgorithmSeqV2::CPUMiningAlgorithmSeqV2() :
	CPUMiningAlgorithmSeq()
{
}


CPUMiningAlgorithmSeqV2::~CPUMiningAlgorithmSeqV2()
{

}
