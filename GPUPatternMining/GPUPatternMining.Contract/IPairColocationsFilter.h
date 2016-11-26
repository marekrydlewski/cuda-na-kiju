#pragma once

#include <memory>
//-------------------------------------------------------------------


class IPairColocationsFilter
{
public:

	virtual void filterPairColocations() = 0;

	virtual ~IPairColocationsFilter()
	{
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<IPairColocationsFilter> IPairColocationsFilterPtr;
//-------------------------------------------------------------------

