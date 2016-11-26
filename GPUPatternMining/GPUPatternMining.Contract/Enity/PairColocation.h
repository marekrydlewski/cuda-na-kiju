#pragma once

__declspec(align(16)) struct PairColocation
{
	int firstType;
	int firstId;
	int secondType;
	int secondId;
};