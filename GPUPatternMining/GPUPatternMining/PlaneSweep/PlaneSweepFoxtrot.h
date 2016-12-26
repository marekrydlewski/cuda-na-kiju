#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../Common/MiningCommon.h"

#include "../Entities/NeighboursListInfoHolder.h"
#include "../Common/CommonOperations.h"


namespace PlaneSweep
{
namespace Foxtrot
{
	/*
	@params

	xCoords				- table with instances x coordinates in ascending order
	yCoords				- table with instances y coordinates
	types				- table with instances types
	instances			- table with feature instances
	count				- count of instances
	radius				- maximal distance between two instances for concidering them as neigbours (inclusive <=) - value for comparision in x-axis
	radiusSquared		- square value of radius - value for comparision cartesian distances between instances
	warpsCount			- more warps means less values to iterate in for loop per warp

	@result

	resultNeighboursCount	- result table that contains number of neigbours for each feature instance

	@info

	tables are treated as zipped (tables values for specified id belong to one instance)
	*/
	__global__ void countNeighbours(
		float* xCoords
		, float* yCoords
		, FeatureInstance* instances
		, int count
		, float radius
		, float radiusSquared
		, int warpsCount
		, UInt* resultNeighboursCount
	);
	// --------------------------------------------------------------------------------------------------------------------------------------


	/*
	@params

	xCoords				- table with instances x coordinates in ascending order
	yCoords				- table with instances y coordinates
	types				- table with instances types
	instances			- table with feature instances
	count				- count of instances
	radius				- maximal distance between two instances for concidering them as neigbours (inclusive <=) - value for comparision in x-axis
	radiusSquared		- square value of radius - value for comparision cartesian distances between instances
	warpsCount			- more warps means less values to iterate in for loop per warp
	outStarts			- result from cuntNeighbours method

	@result

	out_a				- first values of pairs
	out_b				- first values of pairs

	@info

	tables are treated as zipped (tables values for specified id belong to one instance)
	*/
	__global__ void findNeighbours(
		float* xCoords
		, float* yCoords
		, FeatureInstance* instances
		, int count
		, float radius
		, float radiusSquared
		, int warpsCount
		, UInt *outStarts
		, FeatureInstance* out_a
		, FeatureInstance* out_b
	);
	// --------------------------------------------------------------------------------------------------------------------------------------

	/*
	@template

	Txy					- type of coordinates values
	Idx					- type of feature instances
	Ptx					- type of container that holds information about neighbours list

	@params

	xCoords				- table with instances x coordinates in ascending order
	yCoords				- table with instances y coordinates
	instances			- table with instances
	count				- count of instances
	distanceTreshold	- maximal distance between two instances for concidering them as neigbours (inclusive <=)

	@results

	hashMapper			- type pair map to list to those types pairs colocations instances that stores result

	@info

	tables are treated as zipped (tables values for specified id belong to one instance)
	*/
	__host__ void PlaneSweep(
		thrust::device_vector<float> xCoords
		, thrust::device_vector<float>& yCoords
		, thrust::device_vector<FeatureInstance>& instances
		, UInt count
		, float distanceTreshold
		, std::shared_ptr<GPUHashMapper<UInt, NeighboursListInfoHolder, GPUKeyProcessor<UInt>>>& resultHashMap
		, thrust::device_vector<FeatureInstance>& resultPairsA
		, thrust::device_vector<FeatureInstance>& resultPairsB
	);
	// --------------------------------------------------------------------------------------------------------------------------------------

	}
}
