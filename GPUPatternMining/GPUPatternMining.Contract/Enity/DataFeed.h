#pragma once

#include <memory>
#include <vector>
//-------------------------------------------------------------------

struct Coords
{
	float x;
	float y;

	Coords() {}
	Coords(float x, float y) : x(x), y(y) {}
};
//-------------------------------------------------------------------

// ONE instance of DataFeed contains ALL given data  
struct DataFeed
{
	unsigned short type;
	unsigned short instanceId;
	Coords xy;

	DataFeed() {}

	bool operator < (const DataFeed& str) const
	{
		/// smaller type always first e.g A < B, when equal smaller instance id first
		return (type == str.type)? (instanceId < str.instanceId):(type < str.type);
	}

	bool operator > (const DataFeed& str) const
	{
		/// smaller type always first e.g A < B, when equal smaller instance id first
		return (type == str.type) ? (instanceId > str.instanceId) : (type > str.type);
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<DataFeed> DataFeedPtr;
//-------------------------------------------------------------------

template< typename T >
struct array_deleter
{
	void operator ()(T const * p)
	{
		delete[] p;
	}
};

struct ColocationElem
{
	unsigned short type;
	unsigned short instanceId;

	ColocationElem() {};

	ColocationElem(unsigned short type, unsigned short instanceId): type(type), instanceId(instanceId){}

	bool operator < (const ColocationElem& str) const
	{
		/// smaller type always first e.g A < B, when equal smaller instance id first
		return (type == str.type) ? (instanceId < str.instanceId) : (type < str.type);
	}

	bool operator > (const ColocationElem& str) const
	{
		/// smaller type always first e.g A < B, when equal smaller instance id first
		return (type == str.type) ? (instanceId > str.instanceId) : (type > str.type);
	}
};