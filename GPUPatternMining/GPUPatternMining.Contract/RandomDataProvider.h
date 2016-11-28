#pragma once
#include "IDataProvider.h"
#include <random>

class RandomDataProvider :
	public IDataProvider
{
private:
	unsigned int x;
	unsigned int y;
	unsigned int numberOfTypes;
public:
	RandomDataProvider();

	void setRange(unsigned int x, unsigned int y)
	{
		x = x;
		y = y;
	}
	void setNumberOfTypes(unsigned int numberOfTypes)
	{
		numberOfTypes = numberOfTypes;
	}

	DataFeed* getData(size_t s);
	virtual ~RandomDataProvider();
};

