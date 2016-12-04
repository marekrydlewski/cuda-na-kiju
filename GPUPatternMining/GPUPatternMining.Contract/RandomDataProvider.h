#pragma once
#include "IDataProvider.h"

class RandomDataProvider :
	public IDataProvider
{
private:
	unsigned int x;
	unsigned int y;
	unsigned int numberOfTypes;
public:
	RandomDataProvider();

	void setRange(unsigned int x, unsigned int y);

	void setNumberOfTypes(unsigned int numberOfTypes);


	DataFeed* getData(size_t s);
	virtual ~RandomDataProvider();
};

