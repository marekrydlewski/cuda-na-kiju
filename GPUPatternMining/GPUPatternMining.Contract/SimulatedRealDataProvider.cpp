#include "SimulatedRealDataProvider.h"
#include "ApplicationHelper.h"

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData()
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + "\\Resource Files\\fastTestData.txt");
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}
