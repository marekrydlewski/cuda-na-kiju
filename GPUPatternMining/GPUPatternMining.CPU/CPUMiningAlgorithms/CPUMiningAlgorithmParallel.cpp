#include "CPUMiningAlgorithmParallel.h"
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

	insTable = threadInsTable.combine([](
		std::map<unsigned int,
			std::map<unsigned int,
				std::map<unsigned int,
					std::vector<unsigned int>*>>> left,
		std::map<unsigned int, 
			std::map<unsigned int, 
				std::map<unsigned int,
					std::vector<unsigned int>*>>> right
		) ->
			std::map<unsigned int, 
				std::map<unsigned int,
					std::map<unsigned int,
						std::vector<unsigned int>*>>> {
			for (auto it = right.begin(); (it != right.end()); ++it)
			{
				//damn, hard.
				//note: add to left (as left is returned)
			}
			return left;
		}
	);

}

void CPUMiningAlgorithmParallel::filterByPrevalence(float prevalence)
{
}

void CPUMiningAlgorithmParallel::constructMaximalCliques()
{
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


	//radix sort or maybe parallel_sort?
	//https://msdn.microsoft.com/en-us/library/dd470426.aspx#parallel_sorting
	concurrency::parallel_radixsort(M.begin(), M.end());
	concurrency::parallel_radixsort(T.begin(), T.end());
	concurrency::parallel_radixsort(K.begin(), K.end());

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