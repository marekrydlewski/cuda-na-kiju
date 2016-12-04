#pragma once

#include <memory>
#include"Enity\DataFeed.h"
//-------------------------------------------------------------------


class IPairColocationsFilter
{
public:

	virtual void FilterPairColocations(DataFeed* data) = 0;

	virtual ~IPairColocationsFilter()
	{
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<IPairColocationsFilter> IPairColocationsFilterPtr;
//-------------------------------------------------------------------

