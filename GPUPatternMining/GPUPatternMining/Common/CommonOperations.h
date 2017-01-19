#pragma once

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

#include "../HashMap/gpuutils.h"

#include "../HashMap/gpuhashmapper.h"
#include "../Entities/NeighboursListInfoHolder.h"
#include "../HashMap/gpuhashmapper.h"
#include "../Entities/FeatureInstance.h"
#include "../Entities/InstanceTable.h"

//-------------------------------------------------------------------------------

typedef GPUKeyProcessor<unsigned int> GPUUIntKeyProcessor;
typedef HashMapperBean<unsigned int, NeighboursListInfoHolder, GPUUIntKeyProcessor> GPUFeatureInstanceHashMapBean;
//-------------------------------------------------------------------------------


namespace MiningCommon
{
	template<class T>
	__device__ void intraWarpReduce(volatile T *data)
	{
		volatile T* a = data;
		int threadInWarp = threadIdx.x;//&0x1f;

		/*
			binary sum performed by sequence of odd warp threads
		*/
		if ((threadInWarp & 0x01) == 0x01) a[threadInWarp] += a[threadInWarp - 1];
		if ((threadInWarp & 0x03) == 0x03) a[threadInWarp] += a[threadInWarp - 2];
		if ((threadInWarp & 0x07) == 0x07) a[threadInWarp] += a[threadInWarp - 4];
		if ((threadInWarp & 0x0f) == 0x0f) a[threadInWarp] += a[threadInWarp - 8];
		if ((threadInWarp & 0x1f) == 0x1f) a[threadInWarp] += a[threadInWarp - 16];
	}
	//---------------------------------------------------------------------------------------------


	template<class T>
	__device__ void intraWarpScan(volatile T* data)
	{
		volatile T* a = data;
		T temp;
		int threadInWarp = threadIdx.x;//&0x1f;

		//Phase1
		/*
		binary sum performed by sequence of odd warp threads
		*/
		if ((threadInWarp & 0x01) == 0x01) a[threadInWarp] += a[threadInWarp - 1];
		if ((threadInWarp & 0x03) == 0x03) a[threadInWarp] += a[threadInWarp - 2];
		if ((threadInWarp & 0x07) == 0x07) a[threadInWarp] += a[threadInWarp - 4];
		if ((threadInWarp & 0x0f) == 0x0f) a[threadInWarp] += a[threadInWarp - 8];
		//if ((threadIdx.x&0x1f)==0x1f) a[threadInWarp]+=a[threadInWarp-16];

		// TODO check if this is realy important, cause we can just use = 9 lines below
		// zeroing every warp with id = n * 31 where n is any positive total number
		if ((threadInWarp & 0x1f) == 0x1f) a[threadInWarp] = 0;

		//Phase2
		// every warp with id = n * 31 where n is any positive total number do :
		if ((threadIdx.x & 0x1f) == 0x1f)
		{
			temp = a[threadInWarp];
			a[threadInWarp] += a[threadInWarp - 16];
			a[threadInWarp - 16] = temp;
		}

		// TODO check if __syncThread is mandatory here

		if ((threadIdx.x & 0x0f) == 0x0f)
		{
			temp = a[threadInWarp];
			a[threadInWarp] += a[threadInWarp - 8];
			a[threadInWarp - 8] = temp;
		}
		if ((threadIdx.x & 0x07) == 0x07)
		{
			temp = a[threadInWarp];
			a[threadInWarp] += a[threadInWarp - 4];
			a[threadInWarp - 4] = temp;
		}
		if ((threadIdx.x & 0x03) == 0x03)
		{
			temp = a[threadInWarp];
			a[threadInWarp] += a[threadInWarp - 2];
			a[threadInWarp - 2] = temp;
		}
		if ((threadIdx.x & 0x01) == 0x01)
		{
			temp = a[threadInWarp];
			a[threadInWarp] += a[threadInWarp - 1];
			a[threadInWarp - 1] = temp;
		}
	}
	//---------------------------------------------------------------------------------------------

	template<class Tx>
	__device__ float distance(Tx x1, Tx y1, Tx x2, Tx y2)
	{
		return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
	}
	//---------------------------------------------------------------------------------------------

	struct FeatureInstanceComparator
	{
		__host__ __device__  bool operator()(
			const  thrust::tuple<FeatureInstance, FeatureInstance>& o1
			, const thrust::tuple<FeatureInstance, FeatureInstance>& o2)
		{
			if (o1.get<0>().fields.featureId < o2.get<0>().fields.featureId)
				return true;

			if (o1.get<0>().fields.featureId == o2.get<0>().fields.featureId)
			{
				if (o1.get<1>().fields.featureId < o2.get<1>().fields.featureId)
					return true;

				if (o1.get<1>().fields.featureId == o2.get<1>().fields.featureId)
				{
					if (o1.get<0>().fields.instanceId < o2.get<0>().fields.instanceId)
						return true;

					if (o1.get<0>().fields.instanceId == o2.get<0>().fields.instanceId)
					{
						return o1.get<1>().fields.instanceId < o2.get<1>().fields.instanceId;
					}
				}
			}

			return false;
		}
	};
	// --------------------------------------------------------------------------------------------------------------------------------------

	//template<typename C>//, typename TComparator>
	void zipSort(thrust::device_vector<FeatureInstance>& a, thrust::device_vector<FeatureInstance>& b);
	//---------------------------------------------------------------------------------------------

	template <typename T>
	struct InstanceEquality : public thrust::binary_function<T, T, bool>
	{
		__host__ __device__ bool operator()(const T& lhs, const T& rhs) const
		{
			return lhs == rhs;
		}
	};
	//---------------------------------------------------------------------------------------------

	template <typename T>
	struct FirstIndexAndCount
	{
		typedef typename thrust::tuple<T, T> Tuple;

		__host__ __device__
			Tuple operator()(const Tuple& a, const Tuple& b)
		{
			return Tuple(thrust::min(thrust::get<0>(a), thrust::get<0>(b)),
				thrust::get<1>(a) + thrust::get<1>(b));
		}
	};
	//---------------------------------------------------------------------------------------------

	__global__ void InsertIntoHashMap(
		HashMapperBean<unsigned int, NeighboursListInfoHolder, GPUUIntKeyProcessor> bean,
		FeatureInstance* keys,
		unsigned int* deltas,
		unsigned int* counts,
		unsigned int count
	);
	//---------------------------------------------------------------------------------------------

	__global__ void InsertIntoHashMap(
		HashMapperBean<unsigned int, Entities::InstanceTable, GPUUIntKeyProcessor> bean,
		FeatureInstance* keys,
		unsigned int* deltas,
		unsigned int* counts,
		unsigned int count
	);
	//---------------------------------------------------------------------------------------------
}