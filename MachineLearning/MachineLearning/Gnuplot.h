#pragma once

#include <string>

namespace Utils
{

	class Gnuplot
	{
	public:
		enum class ChartType : int
		{
			linearRegression,
			logisticRegression
		};

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

		void setType(ChartType t)
		{
			ctype = t;
		}

	protected:
		ChartType ctype = ChartType::linearRegression;
		std::string relPath = "../../data/";
		std::string dataFileName = "data.txt";
		std::string cmdFileName = "plot.plt";
	};

}

