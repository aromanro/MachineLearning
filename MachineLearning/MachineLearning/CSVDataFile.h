#pragma once

#include "DataFileBase.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <tuple>

namespace Utils {

	class CSVDataFile : public DataFileBase
	{
	public:
		bool Open()
		{
			const std::string name = getFilePath();
			file.open(name);
			return file.is_open();
		}

		std::vector<std::string> readLine()
		{
			std::vector<std::string> res;
			
			std::string line;
			std::getline(file, line);

			std::stringstream lineStream(line);
			std::string field;
			while (std::getline(lineStream, field, ','))
				res.push_back(field);
			
			return res;
		}

	protected:
		std::ifstream file;
	};

	class IrisDataset : public CSVDataFile {
	public:
		typedef std::tuple<double, double, double, double, std::string> Record;

		Record getRecord()
		{
			const std::vector<std::string> r = readLine();
			if (r.size() < 5) return std::make_tuple(0., 0., 0., 0., std::string());

			return std::make_tuple(std::stod(r[0]), std::stod(r[1]), std::stod(r[2]), std::stod(r[3]), r[4]);
		}

		std::vector<Record> getAllRecords()
		{
			std::vector<Record> res;
			while (!file.eof())
			{
				const Record record = getRecord();
				if (std::get<4>(record).empty()) break;
				res.emplace_back(record);
			}
			return res;
		}
	};

}
