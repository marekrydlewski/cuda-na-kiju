#include "IntanceTablesMapCreator.h"
#include "../Entities/InstanceTable.h"
#include "../Common/CommonOperations.h"


namespace IntanceTablesMapCreator
{
	typedef thrust::tuple<FeatureInstance, FeatureInstance> FeatureInstanceTuple;

	typedef thrust::device_vector<FeatureInstance>::iterator FeatureDeviceVectorIterator;
	typedef thrust::tuple<FeatureDeviceVectorIterator, FeatureDeviceVectorIterator> FeatureInstanceIteratorTuple;
	typedef thrust::zip_iterator<FeatureInstanceIteratorTuple> FeatureInstanceTupleIterator;
	// --------------------------------------------------------------------------------------------------------------------------------------

	struct TypesOfFeatureInstancesTypesEquality : public thrust::binary_function<FeatureInstanceTuple, FeatureInstanceTuple, bool>
	{
		__host__ __device__ bool operator()(const FeatureInstanceTuple& lhs, const FeatureInstanceTuple& rhs) const
		{
			return lhs.get<0>().fields.featureId == rhs.get<0>().fields.featureId
				&& lhs.get<1>().fields.featureId == rhs.get<1>().fields.featureId;
		}
	};
	//---------------------------------------------------------------------------------------------

	__global__ void InsertFeatureInstanceTupleIntoHashMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean,
		FeatureInstanceTuple* keys,
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
				(keys[tid].get<0>().field & 0xFFFF0000) | (keys[tid].get<1>().fields.featureId),
				Entities::InstanceTable(counts[tid], deltas[tid])
			);
		}
	}
	//---------------------------------------------------------------------------------------------


	ITMPackPtr createTypedNeighboursListMap(
		thrust::device_vector<FeatureInstance> pairsA
		, thrust::device_vector<FeatureInstance> pairsB
	)
	{
		ITMPackPtr result = std::make_shared<ITMPack>();

		FeatureInstanceTupleIterator zippedBegin = thrust::make_zip_iterator(thrust::make_tuple(
			pairsA.begin()
			, pairsB.begin()
		));

		FeatureInstanceTupleIterator zippedEnd = thrust::make_zip_iterator(thrust::make_tuple(
			pairsA.end()
			, pairsB.end()
		));

		const unsigned int maxResultSize = pairsA.size();

		result->uniques = thrust::device_vector<FeatureInstanceTuple>(maxResultSize);
		result->begins = thrust::device_vector<unsigned int>(maxResultSize);
		result->counts = thrust::device_vector<unsigned int>(maxResultSize);

		result->count = thrust::reduce_by_key(
			zippedBegin,
			zippedEnd,
			thrust::make_zip_iterator(
				thrust::make_tuple(
					thrust::counting_iterator<unsigned int>(0),
					thrust::constant_iterator<unsigned int>(1)
				)
			),
			result->uniques.begin(),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					result->begins.begin(),
					result->counts.begin()
				)
			),
			TypesOfFeatureInstancesTypesEquality(),
			MiningCommon::FirstIndexAndCount<unsigned int>()
		).first - result->uniques.begin();

		constexpr float entryCountHashMapMultiplier = 1.5f;
		result->map.reset(new GPUHashMapper<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor>(
			result->count * entryCountHashMapMultiplier,
			new  GPUUIntKeyProcessor())
		);

		dim3 insertGrid;
		findSmallest2D(result->count, 256, insertGrid.x, insertGrid.y);

		InsertFeatureInstanceTupleIntoHashMap <<< insertGrid, 256 >>>(
			result->map->getBean(),
			thrust::raw_pointer_cast(result->uniques.data())
			, thrust::raw_pointer_cast(result->begins.data())
			, thrust::raw_pointer_cast(result->counts.data()),
			result->count
		);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		return result;
	}
}

