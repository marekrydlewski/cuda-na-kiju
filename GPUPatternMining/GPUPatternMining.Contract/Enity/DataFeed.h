#pragma once

#include <memory>
//-------------------------------------------------------------------

struct Coords
{
	float x;
	float y;

	Coords(float x, float y) : x(x), y(y) {}
};
//-------------------------------------------------------------------

// ONE instance of DataFeed contains ALL given data  
struct DataFeed
{
	int type;
	int instanceId;
	Coords* xy;
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