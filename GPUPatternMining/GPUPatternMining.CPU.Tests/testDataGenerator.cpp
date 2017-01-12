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

DataFeed * TestDataGenerator::getDataForPrevalenceTests()
{
	DataFeed* data = new DataFeed[6];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 0;
	data[1].instanceId = 1;
	data[1].xy.x = 0;
	data[1].xy.y = 10;
	data[2].type = 0;
	data[2].instanceId = 2;
	data[2].xy.x = 0;
	data[2].xy.y = 20;
	data[3].type = 0;
	data[3].instanceId = 3;
	data[3].xy.x = 0;
	data[3].xy.y = 30;
	data[4].type = 1;
	data[4].instanceId = 0;
	data[4].xy.x = 5;
	data[4].xy.y = 0;
	data[5].type = 1;
	data[5].instanceId = 1;
	data[5].xy.x = 5;
	data[5].xy.y = 10;

	return data;
}

DataFeed * TestDataGenerator::getDataForMixedPrevalenceResults()
{
	DataFeed* data = new DataFeed[6];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 0;
	data[1].instanceId = 1;
	data[1].xy.x = 0;
	data[1].xy.y = 10;
	data[2].type = 0;
	data[2].instanceId = 2;
	data[2].xy.x = 0;
	data[2].xy.y = 20;
	data[3].type = 1;
	data[3].instanceId = 0;
	data[3].xy.x = 4;
	data[3].xy.y = 0;
	data[4].type = 1;
	data[4].instanceId = 1;
	data[4].xy.x = -4;
	data[4].xy.y = 0;
	data[5].type = 2;
	data[5].instanceId = 0;
	data[5].xy.x = 8;
	data[5].xy.y = 0;
	
	return data;
}

DataFeed * TestDataGenerator::getDataForMaximalCliqueSize2()
{
	DataFeed* data = new DataFeed[2];
	
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 0;
	data[1].xy.y = 5;

	return data;
}

DataFeed * TestDataGenerator::getDataForMaximalCliqueSize1()
{
	DataFeed* data = new DataFeed[2];

	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 0;
	data[1].xy.y = 100;

	return data;
}

DataFeed * TestDataGenerator::getDataForMaximalCliqueSize4()
{
	DataFeed* data = new DataFeed[4];

	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 0;
	data[1].xy.y = 1;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = 1;
	data[2].xy.y = 0;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = 1;
	data[3].xy.y = 1;

	return data;
}

DataFeed * TestDataGenerator::getDataFor2OverlappingCliques()
{
	DataFeed* data = new DataFeed[5];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 3;
	data[1].xy.y = 1;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = 3;
	data[2].xy.y = -1;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = -3;
	data[3].xy.y = 1;
	data[4].type = 4;
	data[4].instanceId = 0;
	data[4].xy.x = -3;
	data[4].xy.y = -1;

	return data;
}

DataFeed * TestDataGenerator::getDataForDistinctCliques()
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
	data[2].xy.x = 100;
	data[2].xy.y = 0;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = 100;
	data[3].xy.y = 1;
	data[4].type = 4;
	data[4].instanceId = 0;
	data[4].xy.x = 101;
	data[4].xy.y = 0;

	return data;
}

DataFeed * TestDataGenerator::getDataForTreeTest()
{
	DataFeed* data = new DataFeed[14];
	data[0].type = 0;
	data[0].instanceId = 0;
	data[0].xy.x = 0;
	data[0].xy.y = 0;
	data[1].type = 1;
	data[1].instanceId = 0;
	data[1].xy.x = 3;
	data[1].xy.y = 0;
	data[2].type = 2;
	data[2].instanceId = 0;
	data[2].xy.x = 3;
	data[2].xy.y = -3;
	data[3].type = 3;
	data[3].instanceId = 0;
	data[3].xy.x = 0;
	data[3].xy.y = -3;
	data[4].type = 0;
	data[4].instanceId = 1;
	data[4].xy.x = 8;
	data[4].xy.y = 0;
	data[5].type = 3;
	data[5].instanceId = 1;
	data[5].xy.x = 8;
	data[5].xy.y = -3;
	data[6].type = 4;
	data[6].instanceId = 0;
	data[6].xy.x = 13;
	data[6].xy.y = 0;
	data[7].type = 2;
	data[7].instanceId = 1;
	data[7].xy.x = 13;
	data[7].xy.y = -3;
	data[8].type = 0;
	data[8].instanceId = 2;
	data[8].xy.x = 18;
	data[8].xy.y = 0;
	data[9].type = 1;
	data[9].instanceId = 1;
	data[9].xy.x = 18;
	data[9].xy.y = -3;
	data[10].type = 4;
	data[10].instanceId = 1;
	data[10].xy.x = 21;
	data[10].xy.y = 0;
	data[11].type = 2;
	data[11].instanceId = 2;
	data[11].xy.x = 21;
	data[11].xy.y = -3;
	data[12].type = 4;
	data[12].instanceId = 2;
	data[12].xy.x = 26;
	data[12].xy.y = 0;
	data[13].type = 4;
	data[13].instanceId = 3;
	data[13].xy.x = 26;
	data[13].xy.y = -3;

	return data;
}

TestDataGenerator::TestDataGenerator()
{
}

TestDataGenerator::~TestDataGenerator()
{
}
