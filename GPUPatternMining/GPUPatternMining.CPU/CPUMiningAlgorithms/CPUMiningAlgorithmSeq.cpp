#include "CPUMiningAlgorithmSeq.h"

#include "../../GPUPatternMining.Contract/CinsTree.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <chrono>

void CPUMiningAlgorithmSeq::loadData(DataFeed * data, size_t size, unsigned short types)
{
	this->typeIncidenceCounter.resize(types, 0);
	this->source.assign(data, data + size);
}

void CPUMiningAlgorithmSeq::filterByDistance(float threshold)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	std::sort(source.begin(), source.end(), [](DataFeed& first, DataFeed& second)
	{
		return (first.xy.x < second.xy.x);
	});

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

	begin = std::chrono::steady_clock::now();

	float effectiveThreshold = std::pow(threshold, 2);

	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		++this->typeIncidenceCounter[(*it1).type];
		for (auto it2 = std::next(it1); (it2 != source.end()); ++it2)
		{
			if (std::abs((*it1).xy.x - (*it2).xy.x) > threshold) break;
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
						insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId] = new std::vector<unsigned short>();
					}
					insTable[(*it1_h).type][(*it2_h).type][(*it1_h).instanceId]->push_back((*it2_h).instanceId);
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
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

void CPUMiningAlgorithmSeq::createSize2ColocationsGraph()
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

void CPUMiningAlgorithmSeq::constructMaximalCliques()
{
	createSize2ColocationsGraph();
	auto degeneracy = size2ColocationsGraph.getDegeneracy();
	for (unsigned short const vertex : degeneracy.second)
	{
		std::vector<unsigned short> neighboursWithHigherIndices = size2ColocationsGraph.getVertexNeighboursOfHigherIndex(vertex);
		std::vector<unsigned short> neighboursWithLowerIndices = size2ColocationsGraph.getVertexNeighboursOfLowerIndex(vertex);
		std::vector<unsigned short> thisVertex = { vertex };

		auto generatedCliques = size2ColocationsGraph.bkPivot(
			neighboursWithHigherIndices,
			thisVertex,
			neighboursWithLowerIndices);

		maximalCliques.insert(maximalCliques.end(), generatedCliques.begin(), generatedCliques.end());
	}

	std::set<std::vector<unsigned short>> tmp(maximalCliques.begin(), maximalCliques.end());
	std::vector<std::vector<unsigned short>> tmpVec(tmp.begin(), tmp.end());
	maximalCliques.swap(tmpVec);
}

std::vector<std::vector<unsigned short>> CPUMiningAlgorithmSeq::filterMaximalCliques(float prevalence)
{
	std::vector<std::vector<unsigned short>> finalMaxCliques;

	std::vector<std::vector<std::vector<unsigned short>>> cliquesToProcess(
		(*std::max_element(
			maximalCliques.begin(), 
			maximalCliques.end(), 
			[] (std::vector<unsigned short>& left, std::vector<unsigned short>& right) 
			{
				return left.size() < right.size();
			})
		).size());

	for (auto& cl : maximalCliques)
	{
		cliquesToProcess[cl.size()-1].push_back(cl);
	}

	for (int i = cliquesToProcess.size() - 1; i >= 1; --i)
	{
		//no need to 'while' here, getPrevalentMaxCliques edits cliquesToProcess but in other rows
		for (auto& clique : cliquesToProcess[i])
		{
			auto maxCliques = getPrevalentMaxCliques(clique, prevalence, cliquesToProcess);

			if (maxCliques.size() != 0)
				finalMaxCliques.insert(finalMaxCliques.end(), maxCliques.begin(), maxCliques.end());
		}
		cliquesToProcess[i].clear();
	}

	//add colocations of size 1
	finalMaxCliques.insert(finalMaxCliques.end(), cliquesToProcess[0].begin(), cliquesToProcess[0].end());

	return finalMaxCliques;
}

bool CPUMiningAlgorithmSeq::filterNodeCandidate(
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


std::map<std::pair<unsigned short, unsigned short>, std::pair<unsigned short, unsigned short>> CPUMiningAlgorithmSeq::countUniqueInstances()
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
std::vector<std::vector<ColocationElem>> CPUMiningAlgorithmSeq::constructCondensedTree(
	const std::vector<unsigned short>& Cm)
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

bool CPUMiningAlgorithmSeq::isCliquePrevalent(std::vector<unsigned short>& clique, float prevalence)
{
	if (clique.size() == 1 || clique.size() == 2) return true;

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

std::vector<std::vector<unsigned short>> CPUMiningAlgorithmSeq::getPrevalentMaxCliques(
	std::vector<unsigned short>& clique,
	float prevalence,
	std::vector<std::vector<std::vector<unsigned short>>>& cliquesToProcess)
{
	std::vector<std::vector<unsigned short>> finalMaxCliques;

	if (!cliquesContainer.checkCliqueExistence(clique)) 
	{
		if (isCliquePrevalent(clique, prevalence))
		{
			finalMaxCliques.push_back(clique);
			cliquesContainer.insertClique(clique);
		}
		else
		{
			if (clique.size() > 2) //it's possible, no idea why
			{
				auto smallerCliques = getAllCliquesSmallerByOne(clique);
				if (smallerCliques[0].size() == 2) //no need to construct tree, already checked by filterByPrevalence
				{
					for (auto smallClique : smallerCliques)
					{
						if (!cliquesContainer.checkCliqueExistence(smallClique))
						{
							finalMaxCliques.push_back(smallClique);
							cliquesContainer.insertClique(smallClique);
						}
					}
				}
				else
				{
					cliquesToProcess[clique.size() - 2].insert(
						cliquesToProcess[clique.size() - 2].end(),
						smallerCliques.begin(),
						smallerCliques.end());
				}
			}
		}
	};
	return finalMaxCliques;
}

CPUMiningAlgorithmSeq::CPUMiningAlgorithmSeq():
	CPUMiningBaseAlgorithm()
{
}


CPUMiningAlgorithmSeq::~CPUMiningAlgorithmSeq()
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
