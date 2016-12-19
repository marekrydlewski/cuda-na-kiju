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
