#include "Tests.h"


bool LinearRegressionTests()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 10.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Gnuplot plot;

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
		GeneralizedLinearModel<double, double, double, AdamSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, IdentityFunction<double>, L2Loss<double>>, Eigen::RowVectorXd> generalLinearModel;
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

		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, AdamSolver<>> generalLinearModel(3, 3);
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
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(2, 1);

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
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, theSolver> generalLinearModel(3, 1);

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

	return true;
}