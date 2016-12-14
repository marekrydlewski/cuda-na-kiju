#pragma once

// TODO describe why FeatureInstance should be union
union __declspec(align(32)) FeatureInstance
{
	unsigned int field;

	struct
	{
		unsigned short int instanceId;
		unsigned short int featureId;
	} fields;
};

struct __declspec(align(64)) FeatureInstancesPair
{
	unsigned short int firstInstanceId;
	unsigned short int firstFeatureId;

	unsigned short int secondInstanceId;
	unsigned short int secondFeatureId;
};