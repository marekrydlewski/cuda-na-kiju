#include "InstanceTypedNeighboursMapCreator.h"

#include "..\Common/CommonOperations.h"

// ----------------------------------------------------------


namespace InstanceTypedNeighboursMapCreator
{
	typedef thrust::tuple<FeatureInstance, FeatureInstance> FeatureInstanceTuple;

	typedef thrust::device_vector<FeatureInstance>::iterator FeatureDeviceVectorIterator;
	typedef thrust::tuple<FeatureDeviceVectorIterator, FeatureDeviceVectorIterator> FeatureInstanceIteratorTuple;
	typedef thrust::zip_iterator<FeatureInstanceIteratorTuple> FeatureInstanceTupleIterator;
	// ---------------------------------------------------------------------------------------------

	__global__ void InsertIntoHashMap(
		HashMapperBean<unsigned long long, NeighboursListInfoHolder, GPUULongIntKeyProcessor> bean,
		FeatureInstanceTuple* keys,
		unsigned int* begins,
		unsigned int* counts,
		unsigned int count
	)
	{
		unsigned int tid = computeLinearAddressFrom2D();

		if (tid < count)
		{
			GPUHashMapperProcedures::insertKeyValuePair(
				bean,
				createITNMKey(keys[tid].get<0>(), keys[tid].get<1>().fields.featureId),
				NeighboursListInfoHolder(counts[tid], begins[tid])
			);
		}
	}
	//---------------------------------------------------------------------------------------------

	struct FirstFeatureInstanceAndSecondTypeEquality : public thrust::binary_function<FeatureInstanceTuple, FeatureInstanceTuple, bool>
	{
		__host__ __device__ bool operator()(const FeatureInstanceTuple& lhs, const FeatureInstanceTuple& rhs) const
		{
			return lhs.get<0>().field == rhs.get<0>().field
				&& lhs.get<1>().fields.featureId == rhs.get<1>().fields.featureId;
		}
	};
	// ---------------------------------------------------------------------------------------------

	ITNMPackPtr createTypedNeighboursListMap(
		thrust::device_vector<FeatureInstance> pairsA
		, thrust::device_vector<FeatureInstance> pairsB
	)
	{
		ITNMPackPtr result = std::make_shared<ITNMPack>();

		// generate map data

		const unsigned int maxSize = pairsA.size();

		result->begins = thrust::device_vector<unsigned int>(maxSize);
		result->counts = thrust::device_vector<unsigned int>(maxSize);

		thrust::device_vector<FeatureInstanceTuple> firstUniqueElements(maxSize);

		FeatureInstanceTupleIterator zippedBegin = thrust::make_zip_iterator(thrust::make_tuple(
			pairsA.begin()
			, pairsB.begin()
		));

		FeatureInstanceTupleIterator zippedEnd = thrust::make_zip_iterator(thrust::make_tuple(
			pairsA.end()
			, pairsB.end()
		));

		result->count = thrust::reduce_by_key(
			zippedBegin,
			zippedEnd,
			thrust::make_zip_iterator(
				thrust::make_tuple(
					thrust::counting_iterator<unsigned int>(0),
					thrust::constant_iterator<unsigned int>(1)
				)
			),
			firstUniqueElements.begin(),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					result->begins.begin(),
					result->counts.begin()
				)
			),
			FirstFeatureInstanceAndSecondTypeEquality(),
			MiningCommon::FirstIndexAndCount<unsigned int>()
		).first - firstUniqueElements.begin();

		cudaDeviceSynchronize();

		// insert into hash map

		constexpr float entryCountHashMapMultiplier = 1.5f;

		result->map.reset(new TypedNeighboursListMap(
			result->count * entryCountHashMapMultiplier,
			new  GPUULongIntKeyProcessor())
		);

		dim3 insertGrid;
		findSmallest2D(result->count, 256, insertGrid.x, insertGrid.y);

		InsertIntoHashMap << <insertGrid, 256 >> >(
			result->map->getBean(),
			thrust::raw_pointer_cast(firstUniqueElements.data())
			, thrust::raw_pointer_cast(result->begins.data())
			, thrust::raw_pointer_cast(result->counts.data()),
			result->count
			);
		
		cudaDeviceSynchronize();

		return result;
	}
}