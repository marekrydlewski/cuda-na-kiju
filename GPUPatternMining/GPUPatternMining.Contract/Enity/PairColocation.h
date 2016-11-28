#pragma once

__declspec(align(16)) struct PairColocation
{
	PairColocation() 
		:firstType(0), firstId(0), secondType(0), secondId(0)
	{
		
	}
	
	PairColocation(int firstType, int firstId, int secondType, int secondId)
		: firstType(firstType), firstId(firstId), secondType(secondType), secondId(secondId)
	{

	}

	bool operator==(const PairColocation& other) const
	{
		return
			firstType == other.firstType
			&& firstId == other.firstId
			&& secondType == other.secondType
			&& secondId == other.secondId;
	}

	int firstType;
	int firstId;
	int secondType;
	int secondId;
};