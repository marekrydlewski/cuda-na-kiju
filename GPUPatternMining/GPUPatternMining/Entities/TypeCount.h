#pragma once

#include <vector>
#include <memory>
// ------------------------------------------------------------------------------



struct TypeCount
{
	TypeCount(unsigned int type) 
		: type(type), count(0)
	{

	}

	TypeCount(unsigned int type, unsigned short count)
		: type(type), count(count)
	{

	}

	unsigned int type;
	unsigned short count;
};
// ------------------------------------------------------------------------------


typedef std::vector<TypeCount> TypesCounts;
typedef std::shared_ptr<TypesCounts> TypesCountsPtr;
// ------------------------------------------------------------------------------