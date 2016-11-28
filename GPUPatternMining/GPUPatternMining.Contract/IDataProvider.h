#pragma once

#include <memory>

#include "Enity\DataFeed.h"
//-------------------------------------------------------------------


class IDataProvider
{
public:
	
	virtual DataFeed* getData(size_t s) = 0;

	virtual ~IDataProvider()
	{
		
	}
};
//-------------------------------------------------------------------

std::shared_ptr<IDataProvider> IDataProviderPtr;
//-------------------------------------------------------------------