#include "SimulatedRealDataProvider.h"

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData()
{
	loader.loadFromTxtFile("Resource Files/test1.txt");
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}
