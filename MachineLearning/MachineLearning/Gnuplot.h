#pragma once

#include <string>

class Gnuplot
{
public:
	void Execute();

	void setRelativePath(const std::string& p)
	{
		relPath = p;
		if (relPath.empty()) return;
		
		if (relPath[relPath.length() - 1] != '/' && relPath[relPath.length() - 1] != '\\')
			relPath += "/";
	}

	void setDataFileName(const std::string& d)
	{
		dataFileName = d;
	}

	void setCmdFileName(const std::string& c)
	{
		cmdFileName = c;
	}

protected:
	std::string relPath = "../../data/";
	std::string dataFileName = "data.txt";
	std::string cmdFileName = "plot.plt";
};

