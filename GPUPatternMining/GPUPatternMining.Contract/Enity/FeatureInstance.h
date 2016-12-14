#pragma once

// TODO describe why FeatureInstance should be union
union __declspec(align(4)) FeatureInstance
{
	uint32_t field;

	struct
	{
		uint16_t instanceId;
		uint16_t featureId;
	} fields;
};

struct __declspec(align(8)) FeatureInstancesPair
{
	FeatureInstance first;
	FeatureInstance second;
};