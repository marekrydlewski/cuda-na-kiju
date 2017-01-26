#include "SimulatedRealDataProvider.h"
#include "ApplicationHelper.h"

const std::map<DataSet, std::string> SimulatedRealDataProvider::datasetNames = {
	{ DataSet::VeryFast, "\\Resource Files\\veryFastData.txt" },
	{ DataSet::Fast, "\\Resource Files\\fastData.txt" },
	{ DataSet::Medium, "\\Resource Files\\mediumData.txt" },
	{ DataSet::Huge, "\\Resource Files\\hugeData.txt" }
};

const std::map<InstanceSizeTestDataSet, std::string> SimulatedRealDataProvider::instanceSizeTestDatasetNames = {
	{ InstanceSizeTestDataSet::Level1, "\\Resource Files\\instanceSizeTest1.txt" },
	{ InstanceSizeTestDataSet::Level2, "\\Resource Files\\instanceSizeTest2.txt" },
	{ InstanceSizeTestDataSet::Level3, "\\Resource Files\\instanceSizeTest3.txt" },
	{ InstanceSizeTestDataSet::Level4, "\\Resource Files\\instanceSizeTest4.txt" },
	{ InstanceSizeTestDataSet::Level5, "\\Resource Files\\instanceSizeTest5.txt" },
	{ InstanceSizeTestDataSet::Level6, "\\Resource Files\\instanceSizeTest6.txt" },
	{ InstanceSizeTestDataSet::Level7, "\\Resource Files\\instanceSizeTest7.txt" },
	{ InstanceSizeTestDataSet::Level8, "\\Resource Files\\instanceSizeTest8.txt" }
};

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData(DataSet dataset)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + datasetNames.at(dataset));
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getInstanceSizeTestData(InstanceSizeTestDataSet dataset)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + instanceSizeTestDatasetNames.at(dataset));
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}

std::tuple<DataFeed*, int, int> SimulatedRealDataProvider::getTestData(std::string fileName)
{
	loader.loadFromTxtFile(ApplicationHelper::getCurrentWorkingDirectory() + "\\Resource Files\\" + fileName + ".txt");
	return std::make_tuple(loader.getData(), loader.getDataSize(), loader.getNumberOfTypes());
}

