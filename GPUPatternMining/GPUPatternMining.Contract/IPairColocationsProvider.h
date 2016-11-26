#pragma once

#include <memory>
//-------------------------------------------------------------------


class IPairColocationsProvider
{
public:

	virtual void* getPairColocations() = 0;

	virtual ~IPairColocationsProvider()
	{
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<IPairColocationsProvider> IPairColocationsProviderPtr;
//-------------------------------------------------------------------