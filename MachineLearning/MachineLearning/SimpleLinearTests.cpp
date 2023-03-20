#include "Tests.h"


bool SimpleLinearRegressionTests()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 10.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	{
		GLM::SimpleLinearRegression simpleLinearRegression;

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
		GLM::MultivariateSimpleLinearRegression<double> multivariateSimpleLinearRegression(1, 3);

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
		GLM::MultivariateSimpleLinearRegression<> multivariateSimpleLinearRegression(3, 3);

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

	return true;
}