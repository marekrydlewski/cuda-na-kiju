#include "CPUMiningAlgorithmParallel.h"

#include "../../GPUPatternMining.Contract/CinsTree.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"

#include <algorithm>
#include <cassert>
#include <ppl.h>
#include <concrtrm.h>
#include <concurrent_unordered_map.h>
#include <concurrent_vector.h>


void CPUMiningAlgorithmParallel::loadData(DataFeed * data, size_t size, unsigned short types)
{
	this->typeIncidenceCounter.resize(types, 0);
	this->source.assign(data, data + size);
}

//imho impossible to do effective parallelisation
void CPUMiningAlgorithmParallel::filterByDistance(float threshold)
{
	float effectiveThreshold = pow(threshold, 2);

	int cores = concurrency::GetProcessorCount();

	std::vector<unsigned int> loadPerProcessor(cores);

	unsigned int divider = std::pow(2, cores - 1);

	//further iterations will have less work (for first item in data you have to go through whole data, for
	//each next one you have to do one data item less)
	for (int i = 0; i < cores; ++i)
	{
		if (i > 1)
			divider /= 2;

		//last thread gets remaining load
		if (i == cores - 1)
		{
			unsigned int tmp = 0;
			for (int j = 0; j < i; ++j) 
			{
				tmp += loadPerProcessor[j];
			}
			loadPerProcessor[i] = source.size() - tmp;
			break;
		}

		loadPerProcessor[i] = source.size() / divider;
	}

	concurrency::combinable<std::map<unsigned short,
		std::map<unsigned short,
		std::map<unsigned short,
		std::vector<unsigned short>*>>>> combinableInsTable;

	concurrency::parallel_for(0, cores, [&](int i) 
	{
		unsigned int startIndex = 0;
		for (int j = 0; j < i; ++j)
		{
			startIndex += loadPerProcessor[j];
		}

		for (auto it1 = source.begin() + startIndex; (it1 != source.begin() + startIndex + loadPerProcessor[i]); ++it1)
		{
			++typeIncidenceCounter[(*it1).type];
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

						if (combinableInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] == nullptr)
						{
							combinableInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] = new std::vector<unsigned short>();
						}
						combinableInsTable.local()[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId]->push_back((*it2_h).instanceId);
					}
				}
			}
		}
	}, concurrency::static_partitioner());
	
	combinableInsTable.combine_each([&](
	std::map<unsigned short, 
		std::map<unsigned short, 
			std::map<unsigned short,
				std::vector<unsigned short>*>>> local) 
	{
		for (auto it1 = local.begin(); (it1 != local.end()); ++it1)
		{
			for (auto it2 = (*it1).second.begin(); (it2 != (*it1).second.end()); ++it2)
			{
				for (auto it3 = (*it2).second.begin(); (it3 != (*it2).second.end()); ++it3)
				{
					if (insTable[(*it1).first][(*it2).first][(*it3).first] == nullptr)
					{
						insTable[(*it1).first][(*it2).first][(*it3).first] = (*it3).second;
					}	
					else
					{
						insTable[(*it1).first][(*it2).first][(*it3).first]->insert(
							insTable[(*it1).first][(*it2).first][(*it3).first]->end(),
								(*(*it3).second).begin(),
								(*(*it3).second).end());
					}	
				}
			}
		}
	});
}


//already parallelised
void CPUMiningAlgorithmParallel::filterByPrevalence(float prevalence)
{
	auto countedInstances = countUniqueInstances();
	//filtering
	//concurrency::parallel_for_each(
	//	insTable.begin(),
	//	insTable.end(),
	//	[&] (auto &a) {
	//		for (auto& b : a.second)
	//		{
	//			auto aType = a.first;
	//			auto bType = b.first;

	//			bool isPrevalence = countPrevalence(
	//				countedInstances[std::make_pair(aType, bType)],
	//				std::make_pair(typeIncidenceCounter[aType], typeIncidenceCounter[bType]), prevalence);

	//			if (!isPrevalence)
	//			{
	//				for (auto& c : b.second)
	//				{
	//					delete c.second;
	//					//clear vectors' memeory firstly
	//				}
	//				insTable[aType][bType].clear();
	//				//clear all keys
	//			}
	//		}
	//});

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

void CPUMiningAlgorithmParallel::createSize2ColocationsGraph()
{
	size2ColocationsGraph.setSize(typeIncidenceCounter.size());

	for (auto& type1Id : insTable)
	{
		for (auto& type2Id : type1Id.second)
		{
			for (auto& type1instanceId : type2Id.second)
			{
				if (type1instanceId.second->size() > 0)
				{
					size2ColocationsGraph.addEdge(type1Id.first, type2Id.first);
					break;
				}
			}
		}
	}
}


//already parallelised
void CPUMiningAlgorithmParallel::constructMaximalCliques()
{
	createSize2ColocationsGraph();
	auto degeneracy = size2ColocationsGraph.getDegeneracy();

	concurrency::combinable<std::vector<std::vector<unsigned short>>> concurrentMaxCliques;

	concurrency::parallel_for_each(
		degeneracy.second.begin(), 
		degeneracy.second.end(),
		[&] (unsigned short vertex ) {
			std::vector<unsigned short> neighboursWithHigherIndices = size2ColocationsGraph.getVertexNeighboursOfHigherIndex(vertex);
			std::vector<unsigned short> neighboursWithLowerIndices = size2ColocationsGraph.getVertexNeighboursOfLowerIndex(vertex);
			std::vector<unsigned short> thisVertexVector = { vertex };

			auto generatedCliques = size2ColocationsGraph.bkPivot(
				neighboursWithHigherIndices,
				thisVertexVector,
				neighboursWithLowerIndices);

			concurrentMaxCliques.local().insert(
				concurrentMaxCliques.local().end(),
				generatedCliques.begin(),
				generatedCliques.end());
			}
	);

	concurrentMaxCliques.combine_each([this](std::vector<std::vector<unsigned short>>& vec) {
		maximalCliques.insert(maximalCliques.end(), vec.begin(), vec.end());
	});

	std::set<std::vector<unsigned short>> tmp(maximalCliques.begin(), maximalCliques.end());
	std::vector<std::vector<unsigned short>> tmpVec(tmp.begin(), tmp.end());
	maximalCliques.swap(tmpVec);
}


//already parallelised
std::vector<std::vector<unsigned short>> CPUMiningAlgorithmParallel::filterMaximalCliques(float prevalence)
{
	concurrency::combinable<std::vector<std::vector<unsigned short>>> concurrentFinalMaxCliques;

	concurrency::parallel_for_each(
		maximalCliques.begin(), 
		maximalCliques.end(),
		[&] (std::vector<unsigned short> clique) {
			auto maxCliques = getPrevalentMaxCliques(clique, prevalence);
			if (maxCliques.size() != 0)
				concurrentFinalMaxCliques.local().insert(concurrentFinalMaxCliques.local().end(), maxCliques.begin(), maxCliques.end());
		}
	);

	std::vector<std::vector<unsigned short>> finalMaxCliques;
	concurrentFinalMaxCliques.combine_each(
		[&finalMaxCliques] (std::vector<std::vector<unsigned short>> vec) {
			finalMaxCliques.insert(finalMaxCliques.end(), vec.begin(), vec.end());
		}
	);
	return finalMaxCliques;
}

bool CPUMiningAlgorithmParallel::filterNodeCandidate(
	unsigned short type,
	unsigned short instanceId,
	std::vector<CinsNode*> const & ancestors)
{
	for (auto nodePtr : ancestors)
	{
		bool isNeighborOfAncestor = false;
		auto candidatesIt = insTable[nodePtr->type][type].find(nodePtr->instanceId);
		if (candidatesIt != insTable[nodePtr->type][type].end())
		{
			for (auto c : *candidatesIt->second)
			{
				if (c == instanceId) { isNeighborOfAncestor = true; break; }
			}
		}
		if (!isNeighborOfAncestor) return false;
	}
	return true;
}

std::map<std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>> CPUMiningAlgorithmParallel::countUniqueInstances()
{
	std::map<std::pair <unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>> typeIncidenceColocations;

	//counting types incidence
	for (auto& a : insTable)
	{
		for (auto& b : a.second)
		{
			auto aType = a.first;
			auto bType = b.first;

			unsigned short aElements = b.second.size();
			unsigned short bElements = 0;

			std::map<unsigned short, bool> inIncidenceColocations;

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

//Cm's size must be greater or equal 2
std::vector<std::vector<ColocationElem>> CPUMiningAlgorithmParallel::constructCondensedTree(const std::vector<unsigned short>& Cm)
{
	CinsTree tree;
	std::vector<std::vector<ColocationElem>> finalInstances;
	//step1
	for (const auto& t : insTable[Cm[0]][Cm[1]])
	{
		auto a = t.first;
		auto foundNode = tree.root->indexChild(a, Cm[0]);
		if (foundNode == nullptr)
			foundNode = tree.root->addChildPtr(a, Cm[0]);

		for (auto b : *t.second)
		{
			auto newChild = foundNode->addChildPtr(b, Cm[1]);
			tree.lastLevelChildren.push_back(newChild);
		}
	}
	//step2
	//only if co-location greater than 2
	if (Cm.size() > 2)
	{
		std::vector<CinsNode*> newLastLevelChildren;
		for (auto i = 2; i < Cm.size(); ++i)
		{
			newLastLevelChildren.clear();
			for (auto lastChildPtr : tree.lastLevelChildren)
			{
				//list El contains candidates for new level of tree
				std::vector<unsigned short> candidateIds, finalCandidatesIds;
				auto ancestors = lastChildPtr->getAncestors();

				//generate candidates based on last leaf
				auto mapIt = insTable[lastChildPtr->type][Cm[i]].find(lastChildPtr->instanceId);
				if (mapIt != insTable[lastChildPtr->type][Cm[i]].end())//if exists such a key
				{
					for (auto b : *mapIt->second)
					{
						candidateIds.push_back(b);
					}
				}

				//obtaining final list
				for (auto el : candidateIds)
				{
					if (filterNodeCandidate(Cm[i], el, ancestors))
					{
						finalCandidatesIds.push_back(el);
					}
				}

				//add last level children, add node
				for (auto el : finalCandidatesIds)
				{
					auto addedNode = lastChildPtr->addChildPtr(el, Cm[i]);
					newLastLevelChildren.push_back(addedNode);
				}
			}
			tree.lastLevelChildren = newLastLevelChildren;
		}
	}

	//return instances, empty when there's any
	for (auto node : tree.lastLevelChildren)
	{
		std::vector<ColocationElem> elems;
		auto path = node->getPath();
		for (auto p : path)
		{
			elems.push_back(ColocationElem(p->type, p->instanceId));
		}
		finalInstances.push_back(elems);
	}
	return finalInstances;
}

bool CPUMiningAlgorithmParallel::isCliquePrevalent(
	std::vector<unsigned short>& clique,
	float prevalence)
{
	if (clique.size() == 1) return true;

	auto maxCliquesIns = constructCondensedTree(clique);

	if (maxCliquesIns.size() != 0)
	{
		//prepare to count prevalence
		std::vector<unsigned short> particularIns, generalIns;
		for (auto ins : maxCliquesIns[0])
		{
			generalIns.push_back(typeIncidenceCounter[ins.type]);
		}

		for (auto i = 0; i < maxCliquesIns[0].size(); ++i)
		{
			//new map for every type, instances are keys
			std::map<unsigned short, bool> isUsed;
			unsigned short insCounter = 0;
			for (auto j = 0; j < maxCliquesIns.size(); ++j)
			{
				if (!isUsed[maxCliquesIns[j][i].instanceId])
				{
					isUsed[maxCliquesIns[j][i].instanceId] = true;
					++insCounter;
				}
			}
			particularIns.push_back(insCounter);
		}
		return countPrevalence(particularIns, generalIns, prevalence);
	}
	return false; //empty
}

std::vector<std::vector<unsigned short>> CPUMiningAlgorithmParallel::getPrevalentMaxCliques(
	std::vector<unsigned short>& clique,
	float prevalence)
{
	std::vector<std::vector<unsigned short>> finalMaxCliques;
	if (isCliquePrevalent(clique, prevalence))
		finalMaxCliques.push_back(clique);
	else
	{
		if (clique.size() > 2) //it's possible, no idea why
		{
			auto smallerCliques = getAllCliquesSmallerByOne(clique);
			for (auto c : smallerCliques)
			{
				if (c.size() == 2) //no need to construct tree, already checked by filterByPrevalence
				{
					finalMaxCliques.insert(finalMaxCliques.end(), smallerCliques.begin(), smallerCliques.end());
					break; //all smallerCliques are the same size, so insert them all and break the loop
				}
				else
				{
					auto nextCliques = getPrevalentMaxCliques(c, prevalence);
					finalMaxCliques.insert(finalMaxCliques.end(), nextCliques.begin(), nextCliques.end());
				}
			}
		}
	}
	return finalMaxCliques;
}

CPUMiningAlgorithmParallel::CPUMiningAlgorithmParallel() :
	CPUMiningBaseAlgorithm()
{
}


CPUMiningAlgorithmParallel::~CPUMiningAlgorithmParallel()
{
	for (auto& a : insTable)
	{
		for (auto& b : a.second)
		{
			for (auto& c : b.second)
			{
				delete c.second;
			}
		}
	}
}
