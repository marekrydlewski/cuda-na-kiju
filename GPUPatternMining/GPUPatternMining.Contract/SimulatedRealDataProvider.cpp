#include "SimulatedRealDataProvider.h"
#include "ApplicationHelper.h"

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData(std::string fileName)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + "\\Resource Files\\" + fileName + ".txt");
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}

