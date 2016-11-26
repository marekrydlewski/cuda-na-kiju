#pragma once

#include <memory>
//-------------------------------------------------------------------

struct Coords
{
	float x;
	float y;
};
//-------------------------------------------------------------------

struct DataFeed
{
	int* type;
	int* instanceId;
	Coords* xy;
};
//-------------------------------------------------------------------

typedef std::shared_ptr<DataFeed> DataFeedPtr;
//-------------------------------------------------------------------