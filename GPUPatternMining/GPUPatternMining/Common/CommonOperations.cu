#include "CommonOperations.h"


namespace MiningCommon
{
	__global__ void InsertIntoHashMap(
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

	void zipSort(thrust::device_vector<FeatureInstance>& a, thrust::device_vector<FeatureInstance>& b)
	{
		typedef thrust::device_ptr<FeatureInstance> FeatureInstanceIterator;
		typedef thrust::tuple<FeatureInstanceIterator, FeatureInstanceIterator> Tcc;
		typedef thrust::zip_iterator<Tcc> OutputZipIterator;

		FeatureInstanceIterator aBegin = a.begin().base();
		FeatureInstanceIterator bBegin = b.begin().base();

		FeatureInstanceIterator aEnd = a.end().base();
		FeatureInstanceIterator bEnd = a.end().base();

		OutputZipIterator begin(thrust::make_tuple(aBegin, bBegin));
		OutputZipIterator end(thrust::make_tuple(aEnd, bEnd));

		thrust::sort(begin, end, FeatureInstanceComparator());
	}
	//---------------------------------------------------------------------------------------------

}