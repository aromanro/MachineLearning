#include "Gnuplot.h"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace Utils
{

	void Gnuplot::Execute()
	{
		std::string cmdPath = getRelativePath();

		{
			std::filesystem::path p = std::filesystem::current_path();
			if (cmdPath[0] != '/' && cmdPath[0] != '\\')
				p += '/';

			p += cmdPath;
			p += getDataFileName();

			if (cmdPath[cmdPath.length() - 1] != '/' && cmdPath[cmdPath.length() - 1] != '\\')
				cmdPath += "/";

			cmdPath += cmdFileName;

			std::ofstream cmdf(cmdPath, std::ios::out | std::ios::trunc);

			if (ctype == ChartType::logisticRegression)
			{
				cmdf << "plot " << p << " index 0 u 1:2 w l lt 1 lw 2 lc rgb \"blue\" title \"Generating Boundary\", " << p << " index 1 u 1:2 w p pt 7 ps 1 lc rgb \"green\" title \"Data Points First Class\", " << p << " index 2 u 1:2 w p pt 7 ps 1 lc rgb \"red\" title \"Data Points Second Class\"" << std::endl;
			}
			else
			{
				cmdf << "plot " << p << " index 0 u 1:2 w l lt 1 lw 2 lc rgb \"blue\" title \"Generating Function\", " << p << " index 1 u 1:2 w p pt 7 ps 1 lc rgb \"green\" title \"Data Points\", " << p << " index 2 u 1:2 w l lt 1 lw 2 lc rgb \"red\" title \"Regression\"" << std::endl;
			}
		}

		std::string cmd = std::string("start /b gnuplot ") + cmdPath;

		system(cmd.c_str());
	}

}
