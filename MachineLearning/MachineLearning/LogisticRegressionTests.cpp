#include "Tests.h"



bool SimpleLogisticRegressionTest()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 10.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Gnuplot plot;

	WeightsInitializerZero initializer;

	// very simple case, every point that's above the line is the '1' class, everything else is the '0' class
	{
		DataFileWriter theFile("../../data/data4.txt");

		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);
		std::vector<double> ovals(nrPoints);

		std::vector<double> fx(nrPoints);
		std::vector<double> fy(nrPoints);

		for (int i = 0; i < nrPoints; ++i)
		{
			xvals[i] = static_cast<double>(i) / 100;
			yvals[i] = (linearFunction(i) + dist(rde)) / 100;
			ovals[i] = (yvals[i] > linearFunction(i) / 100) ? 1. : 0.;

			fx[i] = xvals[i];
			fy[i] = linearFunction(i) / 100;
		}

		theFile.AddDataset(fx, fy);

		LogisticRegression<> logisticModel(2, 1);

		//logisticModel.solver.alpha = 0.01;
		//logisticModel.solver.lim = 100;

		//logisticModel.solver.beta = 0.8;

		logisticModel.solver.alpha = 0.05;
		logisticModel.solver.beta1 = 0.7;
		logisticModel.solver.beta2 = 0.9;
		logisticModel.solver.lim = 2000;

		logisticModel.Initialize(initializer);

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


		std::vector<double> d1x;
		std::vector<double> d2x;
		std::vector<double> d1y;
		std::vector<double> d2y;

		for (int i = 0; i < nrPoints; ++i)
		{
			in(0) = static_cast<double>(i) / 100;
			in(1) = (linearFunction(i) + dist(rde)) / 100;
			res = logisticModel.Predict(in);

			if (res(0) > 0.5)
			{
				d1x.push_back(in(0));
				d1y.push_back(in(1));
			}
			else
			{
				d2x.push_back(in(0));
				d2y.push_back(in(1));
			}
		}
		theFile.AddDataset(d1x, d1y);
		theFile.AddDataset(d2x, d2y);
	}

	plot.setType(Gnuplot::ChartType::logisticRegression);
	plot.setCmdFileName("plot4.plt");
	plot.setDataFileName("data4.txt");
	plot.Execute();

	return true;
}

bool MoreComplexLogisticRegressionTest()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 10.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Gnuplot plot;

	WeightsInitializerZero initializer;

	// a more complex case for the logistic regression, generate a 2d gaussian distribution and pretend that the points inside some radius are one class and the ones outside are the other

	{
		nrPoints = 1000; // looks nicer and it's better with more points than 100

		DataFileWriter theFile("../../data/data5.txt");

		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);
		std::vector<double> ovals(nrPoints);

		std::vector<double> fx(nrPoints);
		std::vector<double> fy(nrPoints);

		// this is done like this because I also want to test the normalizer class
		Normalizer normalizer(2, 1);

		Eigen::MatrixXd x, y;
		x.resize(2, nrPoints);
		y.resize(1, nrPoints);

		const double radius = 10;
		for (int i = 0; i < nrPoints; ++i)
		{
			double xv = dist(rde);
			double yv = dist(rde);
			const double r = sqrt(xv * xv + yv * yv);

			xv += 100.;
			yv += 180.;

			xvals[i] = xv;
			yvals[i] = yv;

			x(0, i) = xv;
			x(1, i) = yv;

			if (r <= radius) ovals[i] = 1.;
			else ovals[i] = 0;

			y(0, i) = ovals[i];

			fx[i] = 100. + cos(2. * M_PI / nrPoints * i) * radius;
			fy[i] = 180. + sin(2. * M_PI / nrPoints * i) * radius;
		}

		theFile.AddDataset(fx, fy);

		normalizer.AddBatch(x, y);

		// now convert the points using the normalizer
		Eigen::MatrixXd avgi = normalizer.getAverageInput();
		Eigen::MatrixXd istdi = normalizer.getVarianceInput().cwiseSqrt().cwiseInverse();

		std::cout << std::endl << "Averages for input:" << std::endl << avgi << std::endl << std::endl;
		std::cout << "Inverse of std for input:" << std::endl << istdi << std::endl << std::endl;

		std::cout << "Averages should be close to 100 and 180 respectively, while 1/std should be aproximately equal with 1/10" << std::endl << std::endl;

		for (int i = 0; i < nrPoints; ++i)
		{
			x.col(i) -= avgi;
			x.col(i) = x.col(i).cwiseProduct(istdi);
		}

		// now convert the data fed into the logistic model to polar coordinates:
		for (int i = 0; i < nrPoints; ++i)
		{
			const double xv = x(0, i);
			const double yv = x(1, i);
			const double r = sqrt(xv * xv + yv * yv);
			const double t = atan2(yv, xv);

			xvals[i] = r;
			yvals[i] = t;
		}


		LogisticRegression<> logisticModel(2, 1);

		logisticModel.solver.alpha = 0.001;
		logisticModel.solver.beta1 = 0.8;
		logisticModel.solver.beta2 = 0.9;
		logisticModel.solver.lim = 200;

		logisticModel.Initialize(initializer);

		const int batchSize = 32;

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		std::uniform_int_distribution<> distIntBig(0, nrPoints - 1);
		for (int i = 0; i <= 100000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				int ind = distIntBig(rde);

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


		std::vector<double> d1x;
		std::vector<double> d2x;
		std::vector<double> d1y;
		std::vector<double> d2y;

		for (int i = 0; i < nrPoints; ++i)
		{
			// generate them in the cartesian coordinates, the same way as before
			in(0) = dist(rde) + 100;
			in(1) = dist(rde) + 180;


			const double savex = in(0);
			const double savey = in(1);

			// normalize them
			in -= avgi;
			in = in.cwiseProduct(istdi);

			// now convert them to cartesian coordinates
			const double xv = in(0);
			const double yv = in(1);
			const double r = sqrt(xv * xv + yv * yv);
			const double t = atan2(yv, xv);

			in(0) = r;
			in(1) = t;

			// predict
			Eigen::VectorXd res = logisticModel.Predict(in);

			if (res(0) > 0.5)
			{
				d1x.push_back(savex);
				d1y.push_back(savey);
			}
			else
			{
				d2x.push_back(savex);
				d2y.push_back(savey);
			}
		}


		theFile.AddDataset(d1x, d1y);
		theFile.AddDataset(d2x, d2y);
	}

	plot.setType(Gnuplot::ChartType::logisticRegression);
	plot.setCmdFileName("plot5.plt");
	plot.setDataFileName("data5.txt");
	plot.Execute();

	return true;
}

bool LogisticRegressionTests()
{
	return SimpleLogisticRegressionTest() && MoreComplexLogisticRegressionTest();
}