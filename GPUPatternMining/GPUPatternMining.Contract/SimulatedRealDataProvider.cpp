#include "SimulatedRealDataProvider.h"
#include "ApplicationHelper.h"

const std::map<DataSet, std::string> SimulatedRealDataProvider::datasetNames = {
	{ DataSet::Fast, "\\Resource Files\\fastTestData.txt" },
	{ DataSet::Medium, "\\Resource Files\\test1.txt" },
	{ DataSet::VeryLarge, "\\Resource Files\\test2.txt" }
};

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData(DataSet dataset)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + datasetNames.at(dataset));
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}
