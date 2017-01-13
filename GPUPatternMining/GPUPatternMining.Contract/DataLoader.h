#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "Enity\DataFeed.h"

class DataLoader
{
public:
	DataLoader() 
	{
	};
	~DataLoader()
	{
		delete data;
	}
	void loadFromTxtFile(std::string path)
	{
		std::vector<unsigned short> types;
		std::map<unsigned int, int> typeCounter;
		std::vector<DataFeed> dataRecords;
		std::ifstream ifstr(path);
		std::string line;

		while (std::getline(ifstr, line))
		{
			auto lineElems = split(line, ';');
			DataFeed record;
			record.type = std::stoi(lineElems[1]);
			record.instanceId = typeCounter[std::stoi(lineElems[1])];
			record.xy.x = std::stof(lineElems[2]);
			record.xy.y = std::stof(lineElems[3]);
			dataRecords.push_back(record);
			if (std::find(types.begin(), types.end(), std::stoi(lineElems[1])) == types.end())
			{
				types.push_back(std::stoi(lineElems[1]));
			}
			++dataSize;
			++typeCounter[std::stoi(lineElems[1])];
		}

		data = new DataFeed[dataSize];
		for (int i = 0; i < dataSize; ++i)
		{
			data[i] = dataRecords[i];
		}
		numberOfTypes = types.size();
	}

	DataFeed* getData()
	{
		return data;
	}

	int getNumberOfTypes()
	{
		return numberOfTypes;
	}

	int getDataSize()
	{
		return dataSize;
	}

private:
	DataFeed* data;
	int numberOfTypes;
	int dataSize;

	void split(const std::string &s, char delim, std::vector<std::string> &elems) {
		std::stringstream ss;
		ss.str(s);
		std::string item;
		while (std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
	}

	std::vector<std::string> split(const std::string &s, char delim) {
		std::vector<std::string> elems;
		split(s, delim, elems);
		return elems;
	}
};