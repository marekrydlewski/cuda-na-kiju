#pragma once

struct __declspec(align(32)) FeatureInstance
{
	unsigned short int instanceId;
	unsigned short int featureId;

	friend bool operator==(const FeatureInstance& a, const FeatureInstance& b);
};

struct __declspec(align(64)) FeatureInstancesPair
{
	unsigned short int firstInstanceId;
	unsigned short int firstFeatureId;

	unsigned short int secondInstanceId;
	unsigned short int secondFeatureId;
};