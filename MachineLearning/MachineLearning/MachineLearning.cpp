// MachineLearning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SimpleLinearRegression.h"
#include "GeneralLinearModel.h"
#include "GradientSolvers.h"
#include "DataFile.h"
#include "GnuPlot.h"


#include <random>

double linearFunction(double x)
{
	return 3. * x + 2.;
}

double linearFunction2(double x)
{
	return 7. * x - 4.;
}

double linearFunction3(double x)
{
	return -2. * x + 1.;
}

double quadraticFunction(double x)
{
	return (x - 45) * (x - 50) / 50.;
}

double polyFunction(double x)
{
	return (x - 5) * (x - 60) * (x - 80) / 1000.;
}


int main()
{
	std::default_random_engine rde(42);

	std::normal_distribution<double> dist(0., 10.);
	
	const unsigned int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Gnuplot plot;

	{
		SimpleLinearRegression simpleLinearRegression;

		Eigen::RowVectorXd x, y;
		x.resize(nrPoints);
		y.resize(nrPoints);

		for (int i = 0; i < nrPoints; ++i)
		{
			x(i) = i;
			y(i) = linearFunction(i) + dist(rde);
		}

		simpleLinearRegression.AddBatch(x, y);

		double loss = simpleLinearRegression.getLoss() / nrPoints;
		std::cout << "Loss: " << loss << std::endl;

		double res = simpleLinearRegression.Predict(7.);

		std::cout << "Prediction for 7 is: " << res << " generating value: " << linearFunction(7) << std::endl;
	}

	
	{
		MultivariateSimpleLinearRegression<double> multivariateSimpleLinearRegression(1, 3);

		Eigen::MatrixXd x, y;
		x.resize(1, nrPoints);
		y.resize(3, nrPoints);

		for (int i = 0; i < nrPoints; ++i)
		{
			x(0, i) = i;
			y(0, i) = linearFunction(i) + dist(rde);
			y(1, i) = linearFunction2(i) + dist(rde);
			y(2, i) = linearFunction3(i) + dist(rde);
		}

		multivariateSimpleLinearRegression.AddBatch(x, y);

		double loss = multivariateSimpleLinearRegression.getLoss() / nrPoints;
		std::cout << "Loss: " << loss << std::endl;

		Eigen::VectorXd res = multivariateSimpleLinearRegression.Predict(12.);

		std::cout << "Prediction for 12 is: (" << res(0) << ", " << res(1) << ", " << res(2) << ") generating value: (" << linearFunction(12) << ", " << linearFunction2(12) << ", " << linearFunction3(12) << ")" << std::endl;
	}
	

	// the above can be also done like this (it's more useful if x values are different for the different simple linear regressions):

	{
		MultivariateSimpleLinearRegression<> multivariateSimpleLinearRegression(3, 3);

		Eigen::MatrixXd x, y;
		x.resize(3, nrPoints);
		y.resize(3, nrPoints);

		for (int i = 0; i < nrPoints; ++i)
		{
			x(0, i) = i;
			y(0, i) = linearFunction(i) + dist(rde);

			x(1, i) = i;
			y(1, i) = linearFunction2(i) + dist(rde);

			x(2, i) = i;
			y(2, i) = linearFunction3(i) + dist(rde);
		}

		multivariateSimpleLinearRegression.AddBatch(x, y);


		double loss = multivariateSimpleLinearRegression.getLoss() / nrPoints;
		std::cout << "Loss: " << loss << std::endl;

		Eigen::VectorXd in(3);
		in(0) = in(1) = in(2) = 12.;
		Eigen::VectorXd res = multivariateSimpleLinearRegression.Predict(in);

		std::cout << "Prediction for 12 is: (" << res(0) << ", " << res(1) << ", " << res(2) << ") generating value: (" << linearFunction(12) << ", " << linearFunction2(12) << ", " << linearFunction3(12) << ")" << std::endl;
	}

	std::cout << std::endl << "With gradient descent: " << std::endl;

	
	{
		DataFileWriter theFile("../../data/data1.txt");
		
		std::vector<int> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);

		// a dataset for the function (for charting):
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = i;
			yvals[i] = linearFunction(i);
		}
		theFile.AddDataset(xvals, yvals);

		// just generate a bunch of values to be used for training
		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = linearFunction(i) + dist(rde);
		theFile.AddDataset(xvals, yvals);

		// a simple linear regression, but with gradient descent
		//GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, GradientDescentSolver<>, Eigen::MatrixXd> generalLinearModel;
		GeneralLinearModel<double, double, double, AdamSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, IdentityFunction<double>, L2Loss<double>>, Eigen::RowVectorXd> generalLinearModel;

		Eigen::RowVectorXd x, y;
		const int batchSize = 32;

		x.resize(batchSize);
		y.resize(batchSize);

		for (int i = 0; i <= 1000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);
				x(b) = xvals[ind];
				y(b) = yvals[ind];
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 500 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}
		const double res = generalLinearModel.Predict(7.);

		std::cout << "Simple linear: Prediction for 7 is: " << res << " generating value: " << linearFunction(7) << std::endl;

		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = generalLinearModel.Predict(xvals[i]);
		theFile.AddDataset(xvals, yvals);
	}

	plot.setCmdFileName("plot1.plt");
	plot.setDataFileName("data1.txt");
	plot.Execute();
	
	{
		std::vector<int> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);
		std::vector<double> yvals2(nrPoints);
		std::vector<double> yvals3(nrPoints);
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = i;
			yvals[i] = linearFunction(i) + dist(rde);
			yvals2[i] = linearFunction2(i) + dist(rde);
			yvals3[i] = linearFunction3(i) + dist(rde);
		}

		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, AdamSolver<>> generalLinearModel(3, 3);

		Eigen::MatrixXd x, y;
		const int batchSize = 32;

		x.resize(3, batchSize);
		y.resize(3, batchSize);

		for (int i = 0; i <= 1000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);
				x(0, b) = xvals[ind];
				y(0, b) = yvals[ind];

				ind = distInt(rde);
				x(1, b) = xvals[ind];
				y(1, b) = yvals2[ind];

				ind = distInt(rde);
				x(2, b) = xvals[ind];
				y(2, b) = yvals3[ind];
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 500 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(3);
		in(0) = in(1) = in(2) = 12.;
		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "General linear: Prediction for 12 is: (" << res(0) << ", " << res(1) << ", " << res(2) << ") generating value: (" << linearFunction(12) << ", " << linearFunction2(12) << ", " << linearFunction3(12) << ")" << std::endl;
	}

	{
		std::vector<int> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);

		DataFileWriter theFile("../../data/data2.txt");


		// a dataset for the function (for charting):
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = i;
			yvals[i] = quadraticFunction(i);
		}
		theFile.AddDataset(xvals, yvals);

		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = quadraticFunction(i) + dist(rde);
		theFile.AddDataset(xvals, yvals);

		//typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, IdentityFunction<Eigen::VectorXd>, L1Loss<Eigen::VectorXd>> theSolver;
		typedef AdamSolver<> theSolver;
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(2, 1);
		generalLinearModel.solver.alpha = 0.001;
		generalLinearModel.solver.beta1 = 0.9;
		generalLinearModel.solver.beta2 = 0.95;
		generalLinearModel.solver.lim = 20;

		Eigen::MatrixXd x, y;
		const int batchSize = 16;

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		for (int i = 0; i <= 1000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);

				x(0, b) = xvals[ind];
				x(1, b) = xvals[ind] * xvals[ind];
				
				y(0, b) = yvals[ind];
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 10000 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(2);
		in(0) = 24.;
		in(1) = 24 * 24.;

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Quadratic: Prediction for 24 is: " << res(0) << " generating value: " << quadraticFunction(24) << std::endl;

		for (int i = 0; i < nrPoints; ++i)
		{
			in(0) = i;
			in(1) = i * i;
			yvals[i] = generalLinearModel.Predict(in)(0);
		}

		theFile.AddDataset(xvals, yvals);
	}


	plot.setCmdFileName("plot2.plt");
	plot.setDataFileName("data2.txt");
	plot.Execute();

	{
		std::vector<int> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);

		DataFileWriter theFile("../../data/data3.txt");

		// a dataset for the function (for charting):
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = i;
			yvals[i] = polyFunction(i);
		}
		theFile.AddDataset(xvals, yvals);

		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = polyFunction(i) + dist(rde);
		theFile.AddDataset(xvals, yvals);

		//typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, IdentityFunction<Eigen::VectorXd>, L1Loss<Eigen::VectorXd>> theSolver;
		typedef AdamSolver<> theSolver;
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(3, 1);

		generalLinearModel.solver.alpha = 0.0001;
		generalLinearModel.solver.beta1 = 0.9;
		generalLinearModel.solver.beta2 = 0.995;
		generalLinearModel.solver.lim = 20;

		//generalLinearModel.solver.alpha = 0.0001;
		//generalLinearModel.solver.beta1 = 0.7;
		//generalLinearModel.solver.beta2 = 0.8;
		//generalLinearModel.solver.lim = 2000;

		Eigen::MatrixXd x, y;
		const int batchSize = 8;

		x.resize(3, batchSize);
		y.resize(1, batchSize);

		for (int i = 0; i <= 10000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);

				x(0, b) = xvals[ind];
				x(1, b) = xvals[ind] * xvals[ind];
				x(2, b) = x(1, b) * xvals[ind];

				y(0, b) = yvals[ind];
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 50000 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(3);
		in(0) = 32.;
		in(1) = 32 * 32.;
		in(2) = in(1) * 32.;

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Quartic: Prediction for 32 is: " << res(0) << " generating value: " << polyFunction(32) << std::endl;

		for (int i = 0; i < nrPoints; ++i)
		{
			in(0) = i;
			in(1) = i * i;
			in(2) = in(1) * i;
			yvals[i] = generalLinearModel.Predict(in)(0);
		}

		theFile.AddDataset(xvals, yvals);
	}

	plot.setCmdFileName("plot3.plt");
	plot.setDataFileName("data3.txt");
	plot.Execute();
}

