// MachineLearning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "SimpleLinearRegression.h"
#include "GeneralLinearModel.h"
#include "GradientSolvers.h"

#include <iostream>

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
	return 3 * x * x - 2. * x + 7.;
}

double poliFunction(double x)
{
	const double x2 = x * x;

	return  2 * x2 - 14. * x + 16;
}


int main()
{
	std::default_random_engine rde(42);

	std::normal_distribution<double> dist(0., 4.);
	
	const unsigned int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);


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
		// a simple linear regression, but with gradient descent
		//GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, GradientDescentSolver<>, Eigen::MatrixXd> generalLinearModel;
		GeneralLinearModel<double, double, double, GradientDescentSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, IdentityFunction<double>, L2Loss<double>>, Eigen::RowVectorXd> generalLinearModel;

		Eigen::RowVectorXd x, y;
		const int batchSize = 32;

		x.resize(batchSize);
		y.resize(batchSize);

		for (int i = 0; i <= 5000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int a = distInt(rde);
				x(b) = a;
				y(b) = linearFunction(a) + dist(rde);
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
	}

	{
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, GradientDescentSolver<>, Eigen::MatrixXd> generalLinearModel(3, 3);

		Eigen::MatrixXd x, y;
		const int batchSize = 16;

		x.resize(3, batchSize);
		y.resize(3, batchSize);

		for (int i = 0; i <= 5000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int a = distInt(rde);
				x(0, b) = a;
				y(0, b) = linearFunction(a) + dist(rde);

				a = distInt(rde);
				x(1, b) = a;
				y(1, b) = linearFunction2(a) + dist(rde);

				a = distInt(rde);
				x(2, b) = a;
				y(2, b) = linearFunction3(a) + dist(rde);
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
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, GradientDescentSolver<>, Eigen::MatrixXd> generalLinearModel(3, 1);

		Eigen::MatrixXd x, y;
		const int batchSize = 16;

		x.resize(3, batchSize);
		y.resize(1, batchSize);

		for (int i = 0; i <= 5000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int a = distInt(rde);
				x(0, b) = 1;
				x(1, b) = a;
				x(2, b) = a * a;
				
				y(0, b) = quadraticFunction(a) + dist(rde);
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 500 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(3);
		in(0) = 1.;
		in(1) = 24.;
		in(2) = 24 * 24.;

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Quadratic: Prediction for 24 is: " << res(0) << " generating value: " << quadraticFunction(24) << std::endl;
	}

	{
		GeneralLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, GradientDescentSolver<>, Eigen::MatrixXd> generalLinearModel(3, 1);

		Eigen::MatrixXd x, y;
		const int batchSize = 16;

		x.resize(3, batchSize);
		y.resize(1, batchSize);

		for (int i = 0; i <= 5000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int a = distInt(rde);

				x(0, b) = 1;
				x(1, b) = a;
				x(2, b) = a * a;

				y(0, b) = poliFunction(a) + dist(rde);
			}

			generalLinearModel.AddBatch(x, y);

			if (i % 500 == 0)
			{
				double loss = generalLinearModel.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;
			}
		}

		Eigen::VectorXd in(4);
		in(0) = 1.;
		in(1) = 64.;
		in(2) = 64 * 64.;

		Eigen::VectorXd res = generalLinearModel.Predict(in);

		std::cout << "Cubic: Prediction for 64 is: " << res(0) << " generating value: " << poliFunction(64) << std::endl;
	}
}

