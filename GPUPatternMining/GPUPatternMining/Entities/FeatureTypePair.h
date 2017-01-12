#pragma once

union __declspec(align(4)) FeatureTypePair
{
	unsigned int combined;

	struct
	{
		unsigned short b;
		unsigned short a;
	} types;
};
//------------------------------------------------------------------------------

__device__ __host__
inline bool operator==(const FeatureTypePair& a, const FeatureTypePair& b)
{
	return a.combined == b.combined;
}
//------------------------------------------------------------------------------

__device__ __host__
inline bool operator!=(const FeatureTypePair& a, const FeatureTypePair& b)
{
	return a.combined != b.combined;
}
//------------------------------------------------------------------------------
