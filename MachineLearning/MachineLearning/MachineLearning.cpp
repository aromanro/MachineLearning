// MachineLearning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SimpleLinearRegression.h"
#include "GeneralLinearModel.h"
#include "GradientSolvers.h"
#include "LogisticRegression.h"

#include "DataFile.h"
#include "GnuPlot.h"


#include <random>
#include <chrono>

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

	std::cout << std::endl << "With gradient descent: " << std::endl << std::endl;

	
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
		generalLinearModel.solver.alpha = 0.03;
		generalLinearModel.solver.beta1 = 0.7;
		generalLinearModel.solver.beta2 = 0.9;
		generalLinearModel.solver.lim = 20;

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
		generalLinearModel.solver.alpha = 0.02;
		generalLinearModel.solver.beta1 = 0.7;
		generalLinearModel.solver.beta2 = 0.9;
		generalLinearModel.solver.lim = 200;

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
		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);

		DataFileWriter theFile("../../data/data2.txt");

		// the division with 100 below is for scaling things down, otherwise the stochastic gradient descent will have a hard time finding the solution
		// normally it will be scaled by standard deviation or the size of the interval, but that should be enough for tests

		// a dataset for the function (for charting):
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = static_cast<double>(i) / 100;
			yvals[i] = quadraticFunction(i) / 100;
		}
		theFile.AddDataset(xvals, yvals);

		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = (quadraticFunction(i) + dist(rde)) / 100;
		theFile.AddDataset(xvals, yvals);

		//typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, IdentityFunction<Eigen::VectorXd>, L1Loss<Eigen::VectorXd>> theSolver;
		//typedef GradientDescentSolver<> theSolver;
		typedef AdamSolver<> theSolver;
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(2, 1);

		generalLinearModel.solver.alpha = 0.02;
		generalLinearModel.solver.beta1 = 0.7;
		generalLinearModel.solver.beta2 = 0.9;
		generalLinearModel.solver.lim = 2000;

		Eigen::MatrixXd x, y;
		const int batchSize = 32;

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		for (int i = 0; i <= 1000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);

				x(0, b) = xvals[ind];
				x(1, b) = xvals[ind] * xvals[ind];
				
				y(0, b) = yvals[ind];
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 100 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(2);
		in(0) = 24. / 100;
		in(1) = 24 * 24. / 10000;

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Quadratic: Prediction for 24 is: " << res(0) * 100 << " generating value: " << quadraticFunction(24) << std::endl;

		for (int i = 0; i < nrPoints; ++i)
		{
			in(0) = xvals[i];
			in(1) = xvals[i] * xvals[i];
			yvals[i] = generalLinearModel.Predict(in)(0);
		}

		theFile.AddDataset(xvals, yvals);
	}


	plot.setCmdFileName("plot2.plt");
	plot.setDataFileName("data2.txt");
	plot.Execute();


	// the division with 100 below is for scaling things down, otherwise the stochastic gradient descent will have a hard time finding the solution
	// normally it will be scaled by standard deviation or the size of the interval, but that should be enough for tests
	
	

	{
		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);

		DataFileWriter theFile("../../data/data3.txt");

		// a dataset for the function (for charting):
		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = static_cast<double>(i) / 100;
			yvals[i] = polyFunction(i) / 100;
		}
		theFile.AddDataset(xvals, yvals);

		for (int i = 0; i < nrPoints; ++i)
			yvals[i] = (polyFunction(i) + dist(rde)) / 100;
		theFile.AddDataset(xvals, yvals);

		//typedef GradientDescentSolver<> theSolver;
		//typedef MomentumSolver<> theSolver;
		//typedef AdaGradSolver<> theSolver;
		//typedef RMSPropSolver<> theSolver;
		
		// for testing with L1 loss
		//typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, IdentityFunction<Eigen::VectorXd>, L1Loss<Eigen::VectorXd>> theSolver;
		
		typedef AdamSolver<> theSolver;
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(3, 1);

		//generalLinearModel.solver.alpha = 0.01;
		//generalLinearModel.solver.lim = 100;

		//generalLinearModel.solver.beta = 0.8;

		generalLinearModel.solver.alpha = 0.01;
		generalLinearModel.solver.beta1 = 0.9;
		generalLinearModel.solver.beta2 = 0.9;
		generalLinearModel.solver.lim = 2000;

		Eigen::MatrixXd x, y;
		const int batchSize = 32;

		x.resize(3, batchSize);
		y.resize(1, batchSize);

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		for (int i = 0; i <= 100000; ++i)
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

			if (i % 10000 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

		std::cout << "Computation took: " << dif / 1000. << " seconds!" << std::endl;

		Eigen::VectorXd in(3);
		in(0) = 32. / 100;
		in(1) = in(0) * in(0);
		in(2) = in(1) * in(0);

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Quartic: Prediction for 32 is: " << res(0) * 100 << " generating value: " << polyFunction(32) << std::endl;

		for (int i = 0; i < nrPoints; ++i)
		{
			in(0) = xvals[i];
			in(1) = xvals[i] * xvals[i];
			in(2) = in(1) * xvals[i];
			yvals[i] = generalLinearModel.Predict(in)(0);
		}

		theFile.AddDataset(xvals, yvals);
	}

	plot.setCmdFileName("plot3.plt");
	plot.setDataFileName("data3.txt");
	plot.Execute();


	std::cout << std::endl << "Logistic regression: " << std::endl << std::endl;


	// very simple case, every point that's above the line is the '1' class, everything else is the '0' class
	{
		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);
		std::vector<double> ovals(nrPoints);

		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = static_cast<double>(i) / 100;
			yvals[i] = (linearFunction(i) + dist(rde)) / 100;
			ovals[i] = (yvals[i] > linearFunction(i) / 100) ? 1. : 0.;
		}

		typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> theSolver;
		LogisticRegression<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> logisticModel(2, 1);

		//logisticModel.solver.alpha = 0.01;
		//logisticModel.solver.lim = 100;

		//logisticModel.solver.beta = 0.8;

		logisticModel.solver.alpha = 0.02;
		logisticModel.solver.beta1 = 0.7;
		logisticModel.solver.beta2 = 0.9;
		logisticModel.solver.lim = 2000;

		Eigen::MatrixXd x, y;
		const int batchSize = 32;

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		
		for (int i = 0; i <= 100000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distInt(rde);

				x(0, b) = xvals[ind];
				x(1, b) = yvals[ind];

				y(0, b) = ovals[ind];
			}

			logisticModel.AddBatch(x, y);

			if (i % 10000 == 0)
			{
				double loss = logisticModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(2);
		in(0) = 32. / 100;
		in(1) = (linearFunction(32) + 3.) / 100.;

		Eigen::VectorXd res = logisticModel.Predict(in);

		std::cout << "Logistic regression: Prediction for a value above is: " << res(0) << " should be > 0.5" << std::endl;

		in(0) = 64. / 100;
		in(1) = (linearFunction(64) - 2.) / 100;
		res = logisticModel.Predict(in);

		std::cout << "Logistic regression: Prediction for a value below is: " << res(0) << " should be < 0.5" << std::endl;
	}
}

