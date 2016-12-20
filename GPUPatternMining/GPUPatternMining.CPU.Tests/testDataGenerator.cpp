#include "testDataGenerator.h"

DataFeed* TestDataGenerator::getNoNeighboursData()
{
	DataFeed* data = new DataFeed[5];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 100;
	data[1].xy.y = 100;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = -100;
	data[2].xy.y = -100;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = -100;
	data[3].xy.y = 100;
	data[4].type = 4;
	data[4].instanceId = 0;
	data[4].xy.x = 100;
	data[4].xy.y = -100;

	return data;
}

DataFeed * TestDataGenerator::getOneNeighbourRelationshipData()
{
	DataFeed* data = new DataFeed[5];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 1;
	data[1].xy.y = 1;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = -100;
	data[2].xy.y = -100;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = -100;
	data[3].xy.y = 100;
	data[4].type = 4;
	data[4].instanceId = 0;
	data[4].xy.x = 100;
	data[4].xy.y = -100;

	return data;
}

DataFeed * TestDataGenerator::getLinearNeighbourRelationshipData()
{
	DataFeed* data = new DataFeed[5];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 0;
	data[1].xy.y = 5;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = 0;
	data[2].xy.y = 10;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = 0;
	data[3].xy.y = 15;
	data[4].type = 4;
	data[4].instanceId = 0;
	data[4].xy.x = 0;
	data[4].xy.y = 20;

	return data;
}

TestDataGenerator::TestDataGenerator()
{
}

TestDataGenerator::~TestDataGenerator()
{
}
