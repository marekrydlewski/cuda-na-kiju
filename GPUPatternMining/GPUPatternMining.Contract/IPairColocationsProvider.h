#pragma once

#include <memory>

#include "Enity/PairColocation.h"
//-------------------------------------------------------------------


class IPairColocationsProvider
{
public:

	virtual PairColocation* getPairColocations() = 0;
	virtual int getPairColocationsCount() = 0;

	virtual ~IPairColocationsProvider()
	{
	}
};
//-------------------------------------------------------------------

typedef std::shared_ptr<IPairColocationsProvider> IPairColocationsProviderPtr;
//-------------------------------------------------------------------