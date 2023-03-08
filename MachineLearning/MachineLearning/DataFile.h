#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

class DataFileWriter
{
public:
	DataFileWriter(const std::string& name);

	template<typename T1, typename T2> bool AddDataset(const std::vector<T1>& x, const std::vector<T2>& y)
	{
		assert(x.size() == y.size());

		if (x.size() != y.size() || !file) return false;

		file << std::endl;

		for (int i = 0; i < x.size(); ++i)
			file << x[i] << " " << y[i] << std::endl;

		file << std::endl;

		return true;
	}

	// add it at the beginning, or starting with an end\n
	void AddPlotCommand(const std::string& cmd)
	{
		file << cmd << std::endl;
	}

protected:
	std::ofstream file;
};

