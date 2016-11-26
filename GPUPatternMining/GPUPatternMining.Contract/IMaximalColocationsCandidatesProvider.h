#pragma once

#include <memory>
//-------------------------------------------------------------------


class IMaximalColocationsCandidatesProvider
{
public:
	
	virtual void* getMaximalColocationsCandidates() = 0;

	virtual ~IMaximalColocationsCandidatesProvider()
	{
	}
};
//-------------------------------------------------------------------

std::shared_ptr<IMaximalColocationsCandidatesProvider> IMaximalColocationsCandidatesProviderPtr;
//-------------------------------------------------------------------
