#pragma once

#include <stdint.h>

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