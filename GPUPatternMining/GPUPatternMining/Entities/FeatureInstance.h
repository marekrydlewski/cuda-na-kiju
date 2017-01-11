#pragma once


// TODO describe why FeatureInstance should be union
union __declspec(align(4)) FeatureInstanceUnion
{
	unsigned int field;

	struct __inner
	{
		unsigned short instanceId;
		unsigned short featureId;
	} fields;
};
//------------------------------------------------------------------------------

typedef FeatureInstanceUnion FeatureInstance;
//------------------------------------------------------------------------------

__device__ __host__
inline bool operator==(const FeatureInstance& a, const FeatureInstance& b)
{
	return a.field == b.field;
}
//------------------------------------------------------------------------------

__device__ __host__
inline bool operator!=(const FeatureInstance& a, const FeatureInstance& b)
{
	return a.field != b.field;
}
//------------------------------------------------------------------------------
