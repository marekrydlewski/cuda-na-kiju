#pragma once

#include <memory>
#include"Enity\DataFeed.h"
//-------------------------------------------------------------------


class IPairColocationsFilter
{
public:
	virtual ~IPairColocationsFilter()
	{
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<IPairColocationsFilter> IPairColocationsFilterPtr;
//-------------------------------------------------------------------

