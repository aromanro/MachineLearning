#pragma once

#include <string>

#include "DataFileBase.h"

namespace Utils
{

	class Gnuplot : public DataFileBase
	{
	public:
		enum class ChartType : int
		{
			linearRegression,
			logisticRegression
		};

		void Execute();

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
		std::string cmdFileName = "plot.plt";
	};

}

