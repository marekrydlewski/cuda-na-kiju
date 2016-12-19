#include "CPUMiningAlgorithmSeq.h"

#include "../../GPUPatternMining.Contract/CinsTree.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include <algorithm>

void CPUMiningAlgorithmSeq::loadData(DataFeed * data, size_t size, unsigned int types)
{
	this->typeIncidenceCounter.resize(types, 0);
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
	for (unsigned int const& vertex : degeneracy.second)
	{
		std::vector<unsigned int> neighboursWithHigherIndices = size2ColocationsGraph.getVertexNeighboursOfHigherIndex(vertex);
		std::vector<unsigned int> neighboursWithLowerIndices = size2ColocationsGraph.getVertexNeighboursOfLowerIndex(vertex);
		std::vector<unsigned int> thisVertexVector = { vertex };
		auto generatedCliques = bkPivot(neighboursWithHigherIndices, thisVertexVector, neighboursWithLowerIndices);
		maximalCliques.insert(maximalCliques.end(), generatedCliques.begin(), generatedCliques.end());
	}
}

void CPUMiningAlgorithmSeq::filterMaximalCliques()
{
	constructCondensedTree(maximalCliques[0]);
}

std::vector<std::vector<unsigned int>> CPUMiningAlgorithmSeq::bkPivot(std::vector<unsigned int> M, std::vector<unsigned int> K, std::vector<unsigned int> T)
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

	for (auto const& vertex : MpivotNeighboursDifference)
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

bool CPUMiningAlgorithmSeq::filterNodeCandidate(unsigned int type, unsigned int instanceId, std::vector<CinsNode*> const & ancestors)
{
	for (auto nodePtr : ancestors)
	{
		bool isNeighborOfAncestor = false;
		auto candidates = insTable[nodePtr->type][type][nodePtr->instanceId];
		if (candidates != nullptr)
		{
			for (auto c : *candidates)
			{
				if (c == instanceId) { isNeighborOfAncestor = true; break; }
			}
		}
		if (!isNeighborOfAncestor) return false;
	}
	return true;
}


///Tomita Tanaka 2006 maximal pivot algorithm
unsigned int CPUMiningAlgorithmSeq::tomitaMaximalPivot(const std::vector<unsigned int>& SUBG, const std::vector<unsigned int>& CAND)
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


void CPUMiningAlgorithmSeq::constructCondensedTree(const std::vector<unsigned int>& Cm)
{
	CinsTree tree;
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
				std::vector<unsigned int>* vec = insTable[Cm[lastChildPtr->type]][Cm[i]][lastChildPtr->instanceId];
				if (vec != nullptr)
				{
					for (auto b : *vec)
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
					auto addedNode = lastChildPtr->addChildPtr(Cm[i], el);
					newLastLevelChildren.push_back(addedNode);
				}
			}
			tree.lastLevelChildren = newLastLevelChildren;
		}
	}

}

CPUMiningAlgorithmSeq::CPUMiningAlgorithmSeq():
	CPUMiningBaseAlgorithm()
{
}


CPUMiningAlgorithmSeq::~CPUMiningAlgorithmSeq()
{
}
