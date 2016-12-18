#include "CommonOperations.cuh"

__global__ void insertIntoHashMap(
	GPUFeatureInstanceHashMapBean bean,
	FeatureInstance* keys,
	unsigned int* deltas,
	unsigned int* counts,
	unsigned int count
)
{
	unsigned int tid = computeLinearAddressFrom2D();

	if (tid < count)
	{
		GPUHashMapperProcedures::insertKeyValuePair(
			bean,
			keys[tid].field,
			NeighboursListInfoHolder(counts[tid], deltas[tid])
		);
	}
}
//---------------------------------------------------------------------------------------------