#pragma once

#include "DataFileBase.h"
#include "TestStatistics.h"

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
		using Record = std::tuple<double, double, double, double, std::string>;

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

		template<class OutClass> static void Get(const Record& record, OutClass& x, int ind = 0)
		{
			x(0, ind) = std::get<0>(record);
			x(1, ind) = std::get<1>(record);
			x(2, ind) = std::get<2>(record);
			x(3, ind) = std::get<3>(record);
		}

		static void Get(const std::vector<Record>& recordSet, Eigen::MatrixXd& x, int i, int ind = 0)
		{
			Get(recordSet[i], x, ind);
		}

		template<class Model> static void PrintStats(const std::vector<Record>& records, int nrOutputs, Model& model)
		{
			TestStatistics setosaStats;
			TestStatistics versicolorStats;
			TestStatistics virginicaStats;

			Eigen::VectorXd in(4);
			Eigen::VectorXd out(nrOutputs);

			long long int correct = 0;
			for (const auto& record : records)
			{
				Get(record, in, 0);

				out(0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
				if (nrOutputs > 1) out(1) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
				if (nrOutputs > 2) out(2) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

				Eigen::VectorXd res = model.Predict(in.col(0));
				setosaStats.AddPrediction(res(0) > 0.5, out(0) > 0.5);

				if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1) > 0.5);
				if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2) > 0.5);

				CountCorrect(res, out, nrOutputs, correct);
			}

			setosaStats.PrintStatistics("Setosa");
			if (nrOutputs > 1) {
				versicolorStats.PrintStatistics("Versicolor");
				if (nrOutputs > 2) virginicaStats.PrintStatistics("Virginica");
			}

			std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(records.size()) << "%" << std::endl << std::endl;
		}



	protected:
		static double getMax(const Eigen::VectorXd& res, int nrOutputs)
		{
			double limp = 0.5;
			for (int j = 0; j < nrOutputs; ++j)
				limp = std::max(limp, res(j));

			return limp;
		}

		static void CountCorrect(const Eigen::VectorXd& res, const Eigen::VectorXd& out, int nrOutputs, long long int& correct)
		{
			const double limp = getMax(res, nrOutputs);

			if (res(0) == limp && out(0) > 0.5) ++correct;
			else if (nrOutputs > 1 && res(1) == limp && out(1) > 0.5) ++correct;
			else if (nrOutputs > 2 && res(2) == limp && out(2) > 0.5) ++correct;
			else if (limp == 0.5 && nrOutputs < 3)
			{
				// all predictions are either 0.5 or less
				bool isCorrect = true;
				if (out(0) > 0.5) isCorrect = false;
				if (nrOutputs > 1 && out(1) > 0.5) isCorrect = false;

				if (isCorrect) ++correct;
			}
		}
	};

}
