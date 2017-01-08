#include "CPUMiningAlgorithmParallel.h"

#include "../../GPUPatternMining.Contract/CinsTree.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"

#include <algorithm>
#include <cassert>
#include <ppl.h>
#include <concurrent_unordered_map.h>
#include <concurrent_vector.h>


void CPUMiningAlgorithmParallel::loadData(DataFeed * data, size_t size, unsigned int types)
{
	this->typeIncidenceCounter.resize(types, 0);
	this->source.assign(data, data + size);
}

//imho impossible to do effective parallelisation
void CPUMiningAlgorithmParallel::filterByDistance(float threshold)
{
	float effectiveThreshold = pow(threshold, 2);


	for (auto it1 = source.begin(); (it1 != source.end()); ++it1)
	{
		++this->typeIncidenceCounter[(*it1).type];
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


//already parallelised
void CPUMiningAlgorithmParallel::filterByPrevalence(float prevalence)
{
	auto countedInstances = countUniqueInstances();
	//filtering
	concurrency::parallel_for_each(
		insTable.begin(),
		insTable.end(),
		[&] (auto &a) {
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
	});

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

	concurrency::combinable<std::vector<std::vector<unsigned int>>> concurrentMaxCliques;

	concurrency::parallel_for_each(
		degeneracy.second.begin(), 
		degeneracy.second.end(),
		[&] (unsigned int vertex ) {
			std::vector<unsigned int> neighboursWithHigherIndices = size2ColocationsGraph.getVertexNeighboursOfHigherIndex(vertex);
			std::vector<unsigned int> neighboursWithLowerIndices = size2ColocationsGraph.getVertexNeighboursOfLowerIndex(vertex);
			std::vector<unsigned int> thisVertexVector = { vertex };
			auto generatedCliques = bkPivot(neighboursWithHigherIndices, thisVertexVector, neighboursWithLowerIndices);
			concurrentMaxCliques.local().insert(concurrentMaxCliques.local().end(), generatedCliques.begin(), generatedCliques.end());
		}
	);

	//why it doesn't work??? - size of vector out of range lol
	//concurrentMaxCliques.combine_each([this](std::vector<std::vector<unsigned int>>& vec) {
	//	maximalCliques.insert(vec.begin(), vec.end(), maximalCliques.end());
	//});

	maximalCliques = concurrentMaxCliques.combine([this](auto& left, auto& right){
		left.insert(right.begin(), right.end(), left.end());
		return left;
	});
}


//already parallelised
std::vector<std::vector<unsigned int>> CPUMiningAlgorithmParallel::filterMaximalCliques(float prevalence)
{
	concurrency::combinable<std::vector<std::vector<unsigned int>>> concurrentFinalMaxCliques;

	concurrency::parallel_for_each(
		maximalCliques.begin(), 
		maximalCliques.end(),
		[&] (std::vector<unsigned int> clique) {
			auto maxCliques = getPrevalentMaxCliques(clique, prevalence);
			if (maxCliques.size() != 0)
				concurrentFinalMaxCliques.local().insert(concurrentFinalMaxCliques.local().end(), maxCliques.begin(), maxCliques.end());
		}
	);

	//why it doesn't work??? - size of vector out of range lol
	//std::vector<std::vector<unsigned int>> finalMaxCliques;
	//concurrentFinalMaxCliques.combine_each(
	//	[&finalMaxCliques] (std::vector<std::vector<unsigned int>> vec) {
	//		finalMaxCliques.insert(vec.begin(), vec.end(), finalMaxCliques.end());
	//	}
	//);
	//return finalMaxCliques;

	return concurrentFinalMaxCliques.combine(
		[] (auto& left, auto& right) {
			left.insert(right.begin(), left.end(), right.end());
			return left;
	});
}

std::vector<std::vector<unsigned int>> CPUMiningAlgorithmParallel::bkPivot(
	std::vector<unsigned int> M,
	std::vector<unsigned int> K,
	std::vector<unsigned int> T)
{
	std::vector<std::vector<unsigned int>> maximalCliques;
	std::vector<unsigned int> MTunion(M.size() + T.size());
	std::vector<unsigned int> MpivotNeighboursDifference(M.size());
	std::vector<unsigned int>::iterator it;

	std::sort(M.begin(), M.end());
	std::sort(T.begin(), T.end());
	std::sort(K.begin(), K.end());

	it = std::set_union(M.begin(), M.end(), T.begin(), T.end(), MTunion.begin());
	MTunion.resize(it - MTunion.begin());

	if (MTunion.size() == 0)
	{
		maximalCliques.push_back(K);
		return maximalCliques;
	}

	unsigned int pivot = tomitaMaximalPivot(MTunion, M);

	auto pivotNeighbours = size2ColocationsGraph.getVertexNeighbours(pivot);
	std::sort(pivotNeighbours.begin(), pivotNeighbours.end());

	it = std::set_difference(M.begin(), M.end(), pivotNeighbours.begin(), pivotNeighbours.end(), MpivotNeighboursDifference.begin());
	MpivotNeighboursDifference.resize(it - MpivotNeighboursDifference.begin());

	for (auto const vertex : MpivotNeighboursDifference)
	{
		std::vector<unsigned int> vertexNeighbours = size2ColocationsGraph.getVertexNeighbours(vertex);
		std::vector<unsigned int> vertexVector = { vertex };
		std::vector<unsigned int> KvertexUnion(K.size() + 1);
		std::vector<unsigned int> MvertexNeighboursIntersection(M.size());
		std::vector<unsigned int> TvertexNeighboursIntersection(T.size());

		std::sort(vertexNeighbours.begin(), vertexNeighbours.end());

		std::set_union(K.begin(), K.end(), vertexVector.begin(), vertexVector.end(), KvertexUnion.begin());

		it = std::set_intersection(M.begin(), M.end(), vertexNeighbours.begin(), vertexNeighbours.end(), MvertexNeighboursIntersection.begin());
		MvertexNeighboursIntersection.resize(it - MvertexNeighboursIntersection.begin());

		it = std::set_intersection(T.begin(), T.end(), vertexNeighbours.begin(), vertexNeighbours.end(), TvertexNeighboursIntersection.begin());
		TvertexNeighboursIntersection.resize(it - TvertexNeighboursIntersection.begin());

		auto generatedCliques = bkPivot(MvertexNeighboursIntersection, KvertexUnion, TvertexNeighboursIntersection);
		maximalCliques.insert(maximalCliques.end(), generatedCliques.begin(), generatedCliques.end());
	}

	return maximalCliques;
}

bool CPUMiningAlgorithmParallel::filterNodeCandidate(
	unsigned int type,
	unsigned int instanceId,
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


///Tomita Tanaka 2006 maximal pivot algorithm
unsigned int CPUMiningAlgorithmParallel::tomitaMaximalPivot(const std::vector<unsigned int>& SUBG, const std::vector<unsigned int>& CAND)
{
	unsigned int u, maxCardinality = 0;
	for (auto& s : SUBG)
	{
		auto neighbors = size2ColocationsGraph.getVertexNeighbours(s);
		std::sort(neighbors.begin(), neighbors.end());
		std::vector<unsigned int> nCANDunion(neighbors.size() + CAND.size());

		auto itUnion = std::set_union(CAND.begin(), CAND.end(), neighbors.begin(), neighbors.end(), nCANDunion.begin());
		nCANDunion.resize(itUnion - nCANDunion.begin());

		if (nCANDunion.size() >= maxCardinality)
		{
			u = s;
			maxCardinality = nCANDunion.size();
		}
	}
	return u;
}

std::map<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> CPUMiningAlgorithmParallel::countUniqueInstances()
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

//Cm's size must be greater or equal 2
std::vector<std::vector<ColocationElem>> CPUMiningAlgorithmParallel::constructCondensedTree(const std::vector<unsigned int>& Cm)
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
				std::vector<unsigned int> candidateIds, finalCandidatesIds;
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

bool CPUMiningAlgorithmParallel::isCliquePrevalent(std::vector<unsigned int>& clique, float prevalence)
{
	if (clique.size() == 1) return true;

	auto maxCliquesIns = constructCondensedTree(clique);

	if (maxCliquesIns.size() != 0)
	{
		//prepare to count prevalence
		std::vector<unsigned int> particularIns, generalIns;
		for (auto ins : maxCliquesIns[0])
		{
			generalIns.push_back(typeIncidenceCounter[ins.type]);
		}

		for (auto i = 0; i < maxCliquesIns[0].size(); ++i)
		{
			//new map for every type, instances are keys
			std::map<unsigned int, bool> isUsed;
			unsigned int insCounter = 0;
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

std::vector<std::vector<unsigned int>> CPUMiningAlgorithmParallel::getPrevalentMaxCliques(std::vector<unsigned int> clique, float prevalence)
{
	std::vector<std::vector<unsigned int>> finalMaxCliques;
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
					finalMaxCliques.insert(finalMaxCliques.end(), smallerCliques.begin(), smallerCliques.end());
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
}
