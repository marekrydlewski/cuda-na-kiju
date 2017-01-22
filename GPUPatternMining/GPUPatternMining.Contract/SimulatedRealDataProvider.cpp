#include "SimulatedRealDataProvider.h"
#include "ApplicationHelper.h"

const std::map<DataSet, std::string> SimulatedRealDataProvider::datasetNames = {
	{ DataSet::VeryFast, "\\Resource Files\\veryFastData.txt" },
	{ DataSet::Fast, "\\Resource Files\\fastData.txt" },
	{ DataSet::Medium, "\\Resource Files\\mediumData.txt" },
	{ DataSet::Huge, "\\Resource Files\\hugeData.txt" }
};

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData(DataSet dataset)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + datasetNames.at(dataset));
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}
