#pragma once

#include <memory>

#include "Entity/DataFeed.h"
//-------------------------------------------------------------------


class IDataProvider
{
public:
	
	virtual DataFeedPtr getData() = 0;

	virtual ~IDataProvider()
	{
		
	}
};
//-------------------------------------------------------------------

std::shared_ptr<IDataProvider> IDataProviderPtr;
//-------------------------------------------------------------------