#pragma once

#include <memory>
//-------------------------------------------------------------------

struct Coords
{
	float x;
	float y;
};
//-------------------------------------------------------------------

// ONE instance of DataFeed contains ALL given data  
struct DataFeed
{
	int* type;
	int* instanceId;
	Coords* xy;
};
//-------------------------------------------------------------------

typedef std::shared_ptr<DataFeed> DataFeedPtr;
//-------------------------------------------------------------------