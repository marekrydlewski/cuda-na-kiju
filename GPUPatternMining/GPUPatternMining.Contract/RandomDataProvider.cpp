#include "RandomDataProvider.h"
#include <vector>

RandomDataProvider::RandomDataProvider()
{
}


DataFeed* RandomDataProvider::getData(size_t s)
{
	std::random_device rdev{};
	std::default_random_engine randomEngine{ rdev() };
	std::uniform_real_distribution<float> disX(0, x);
	std::uniform_real_distribution<float> disY(0, y);
	std::uniform_int_distribution<> disType(1, numberOfTypes);

	std::vector<int> instances(numberOfTypes);

	//DataFeedPtr sp(new DataFeed[s], array_deleter<DataFeed>());
	DataFeed* dataArray = new DataFeed[s];

	for (int i = 0; i < s; ++i)
	{
		DataFeed dataFeed;
		dataFeed.type = disType(rdev);
		dataFeed.xy = Coords(disX(rdev), disY(rdev));
		dataFeed.instanceId = ++instances[dataFeed.type];

		dataArray[i] = dataFeed;
	}

	return dataArray;
}

RandomDataProvider::~RandomDataProvider()
{
}