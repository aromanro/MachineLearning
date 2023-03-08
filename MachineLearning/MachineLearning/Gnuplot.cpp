#include "Gnuplot.h"

#include <fstream>
#include <iostream>
#include <filesystem>

void Gnuplot::Execute()
{
	std::string cmdPath = relPath;
	
	{
		std::ofstream cmdf(cmdPath);
		std::filesystem::path p = std::filesystem::current_path();
		if (cmdPath[0] != '/' && cmdPath[0] != '\\')
			p += '/';

		p += relPath;
		p += dataFileName;

		cmdf << "plot " << p << " index 0 using 1:2 with lines title \"Generating Function\", " << p << " index 1 using 1:2 with points title \"Data Points\", " << p << " index 2 using 1:2 with lines title \"Regression\"" << std::endl;
	}

	cmdPath += cmdFileName;

	std::string cmd = std::string("start /b gnuplot ") + cmdPath;

	system(cmd.c_str());
}