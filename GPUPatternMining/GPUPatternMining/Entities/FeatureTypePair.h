#pragma once

union FeatureTypePair
{
	unsigned int combined;

	struct
	{
		unsigned short a;
		unsigned short b;
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
