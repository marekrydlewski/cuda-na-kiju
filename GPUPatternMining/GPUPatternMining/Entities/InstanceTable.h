#pragma once
#include <host_defines.h>

namespace Entities
{

	class InstanceTable
	{
	public:
		unsigned int count;

		/*
			index in sorted intances table
		*/
		unsigned int startIdx;

		__host__ __device__ InstanceTable()
			: count(0), startIdx(0xffffffff)
		{
		}
		__host__ __device__ InstanceTable
		(unsigned int count, unsigned int index)
			: count(count), startIdx(index)
		{
		}
	};
	// ------------------------------------------------------------

	__device__ __host__
	inline bool operator==(const InstanceTable& a, const InstanceTable& b)
	{
		return a.count == b.count 
			&& a.startIdx == b.startIdx;
	}
	//------------------------------------------------------------------------------

	__device__ __host__
		inline bool operator!=(const InstanceTable& a, const InstanceTable& b)
	{
		return a.count != b.count
			|| a.startIdx != b.startIdx;
	}
	//------------------------------------------------------------------------------
}


