#include "Tests.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"



bool SimpleLogisticRegressionTest()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 30.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Utils::Gnuplot plot;

	Initializers::WeightsInitializerZero initializer;

	// very simple case, every point that's above the line is the '1' class, everything else is the '0' class
	{
		Utils::DataFileWriter theFile("../../data/LogisticRegressionLinear.txt");

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

		GLM::LogisticRegression<> logisticModel(2, 1);

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

	plot.setType(Utils::Gnuplot::ChartType::logisticRegression);
	plot.setCmdFileName("LogisticRegressionLinear.plt");
	plot.setDataFileName("LogisticRegressionLinear.txt");
	plot.Execute();

	return true;
}

bool MoreComplexLogisticRegressionTest()
{
	std::default_random_engine rde(42);
	std::normal_distribution<double> dist(0., 10.);

	// for test points generation, it does not need to be the same distribution as the learning one
	std::normal_distribution<double> distgx(0., 16.);
	std::normal_distribution<double> distgy(0., 8.);

	int nrPoints = 100;

	std::uniform_int_distribution<> distInt(0, nrPoints - 1);

	Utils::Gnuplot plot;

	Initializers::WeightsInitializerZero initializer;

	// a more complex case for the logistic regression, generate a 2d gaussian distribution and pretend that the points inside some radius are one class and the ones outside are the other

	{
		nrPoints = 1000; // looks nicer and it's better with more points than 100

		Utils::DataFileWriter theFile("../../data/LogisticRegressionCircle.txt");

		std::vector<double> xvals(nrPoints);
		std::vector<double> yvals(nrPoints);
		std::vector<double> ovals(nrPoints);

		std::vector<double> fx(nrPoints);
		std::vector<double> fy(nrPoints);

		// this is done like this because I also want to test the normalizer class
		Norm::Normalizer normalizer(2, 1);

		Eigen::MatrixXd x, y;
		x.resize(2, nrPoints);
		y.resize(1, nrPoints);

		const double radius = 10;
		const double centerX = 100;
		const double centerY = 180;

		for (int i = 0; i < nrPoints; ++i)
		{
			double xv = dist(rde);
			double yv = dist(rde);
			const double r = sqrt(xv * xv + yv * yv);

			xv += centerX;
			yv += centerY;

			x(0, i) = xvals[i] = xv;
			x(1, i) = yvals[i] = yv;

			if (r <= radius) ovals[i] = 1.;
			else ovals[i] = 0;

			y(0, i) = ovals[i];

			// the data for drawing the circle for the boundary
			fx[i] = centerX + cos(2. * M_PI / nrPoints * i) * radius;
			fy[i] = centerY + sin(2. * M_PI / nrPoints * i) * radius;
		}

		theFile.AddDataset(fx, fy);

		normalizer.AddBatch(x, y);

		// now convert the points using the normalizer
		const Eigen::MatrixXd avgi = normalizer.getAverageInput();
		const Eigen::MatrixXd istdi = normalizer.getVarianceInput().cwiseSqrt().cwiseInverse();

		std::cout << std::endl << "Averages for input:" << std::endl << avgi << std::endl << std::endl;
		std::cout << "Inverse of std for input:" << std::endl << istdi << std::endl << std::endl;

		std::cout << "Averages should be close to 100 and 180 respectively, while 1/std should be aproximately equal with 1/10 and 1/20 respectively" << std::endl << std::endl;

		// normalize
		for (int i = 0; i < nrPoints; ++i)
		{
			x.col(i) -= avgi;
			x.col(i) = x.col(i).cwiseProduct(istdi);

			// I know them exactly, this is for tests:
			//x(0, i) -= centerX;
			//x(1, i) -= centerY;
			//x.col(i) *= 1. / 10;
		}

		// now convert the data fed into the logistic model to polar coordinates:
		for (int i = 0; i < nrPoints; ++i)
		{
			const double xv = x(0, i);
			const double yv = x(1, i);

			xvals[i] = sqrt(xv * xv + yv * yv);
			yvals[i] = atan2(yv, xv);
		}


		GLM::LogisticRegression<> logisticModel(2, 1);

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
			// generate them in the cartesian coordinates
			// the distribution for test points can be whatever we'd like
			const double xorig = centerX + distgx(rde);
			const double yorig = centerY + distgy(rde);

			in(0) = xorig;
			in(1) = yorig;

			// normalize them
			in -= avgi;
			in = in.cwiseProduct(istdi);

			// I know them exactly, this is for tests:
			//in(0) -= centerX;
			//in(1) -= centerY;
			//in *= 1. / 10;

			// now convert them to cartesian coordinates
			const double xv = in(0);
			const double yv = in(1);

			in(0) = sqrt(xv * xv + yv * yv);
			in(1) = atan2(yv, xv);

			// predict
			Eigen::VectorXd res = logisticModel.Predict(in);

			if (res(0) > 0.5)
			{
				d1x.push_back(xorig);
				d1y.push_back(yorig);
			}
			else
			{
				d2x.push_back(xorig);
				d2y.push_back(yorig);
			}
		}


		theFile.AddDataset(d1x, d1y);
		theFile.AddDataset(d2x, d2y);
	}

	plot.setType(Utils::Gnuplot::ChartType::logisticRegression);
	plot.setCmdFileName("LogisticRegressionCircle.plt");
	plot.setDataFileName("LogisticRegressionCircle.txt");
	plot.Execute();

	return true;
}

bool IrisLogisticRegressionTest()
{
	std::cout << std::endl << "Logistic Regression for the Iris dataset, Setosa is lineary separable from the other two, but the others two cannot be linearly separated, so expect good results for Setosa but not for the other two" << std::endl << std::endl;

	Utils::IrisDataset irisDataset;
	irisDataset.setRelativePath("../../Datasets/");
	irisDataset.setDataFileName("iris.data");

	if (!irisDataset.Open()) return false;

	auto records = irisDataset.getAllRecords();

	const int nrTraining = 100;

	// shuffle the data

	std::random_device rd;
	std::mt19937 g(rd());

	// ensure it's shuffled enough to have all enough samples of all classes in the test set
	for (;;)
	{
		int setosa = 0;
		int versicolor = 0;
		int virginica = 0;

		std::shuffle(records.begin(), records.end(), g);
		std::shuffle(records.begin(), records.end(), g);
		std::shuffle(records.begin(), records.end(), g);

		for (auto it = records.begin() + nrTraining; it != records.end(); ++it)
		{
			const auto rec = *it;
			if (std::get<4>(rec) == "Iris-setosa") ++setosa;
			if (std::get<4>(rec) == "Iris-versicolor") ++versicolor;
			if (std::get<4>(rec) == "Iris-virginica") ++virginica;
		}

		if (setosa > 10 && versicolor > 10 && virginica > 10) break;
	}


	//for (auto rec : records)
	//	std::cout << std::get<0>(rec) << ", " << std::get<1>(rec) << ", " << std::get<2>(rec) << ", " << std::get<3>(rec) << ", " << std::get<4>(rec) << std::endl;

	// split the data into training and test sets


	std::vector<Utils::IrisDataset::Record> trainingSet(records.begin(), records.begin() + nrTraining);
	std::vector<Utils::IrisDataset::Record> testSet(records.begin() + nrTraining, records.end());

	const int nrOutputs = 1; // 1 only for Setosa, 3 if all three classes are to be predicted

	// normalize the inputs
	Norm::Normalizer normalizer(4, nrOutputs);
	Eigen::MatrixXd x(4, nrTraining);
	Eigen::MatrixXd y(nrOutputs, nrTraining);

	for (int i = 0; i < nrTraining; ++i)
	{
		x(0, i) = std::get<0>(trainingSet[i]);
		x(1, i) = std::get<1>(trainingSet[i]);
		x(2, i) = std::get<2>(trainingSet[i]);
		x(3, i) = std::get<3>(trainingSet[i]);

		y(0, i) = (std::get<4>(trainingSet[i]) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) y(1, i) = (std::get<4>(trainingSet[i]) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) y(2, i) = (std::get<4>(trainingSet[i]) == "Iris-virginica") ? 1 : 0;
	}

	normalizer.AddBatch(x, y);

	const Eigen::MatrixXd avgi = normalizer.getAverageInput();
	const Eigen::MatrixXd istdi = normalizer.getVarianceInput().cwiseSqrt().cwiseInverse();

	for (int i = 0; i < nrTraining; ++i)
	{
		x.col(i) -= avgi;
		x.col(i) = x.col(i).cwiseProduct(istdi);

		trainingSet[i] = std::make_tuple(x(0, i), x(1, i), x(2, i), x(3, i), std::get<4>(trainingSet[i]));
	}

	x.resize(4, 1);
	for (int i = 0; i < testSet.size(); ++i)
	{
		x(0, 0) = std::get<0>(testSet[i]);
		x(1, 0) = std::get<1>(testSet[i]);
		x(2, 0) = std::get<2>(testSet[i]);
		x(3, 0) = std::get<3>(testSet[i]);


		x.col(0) -= avgi;
		x.col(0) = x.col(0).cwiseProduct(istdi);

		testSet[i] = std::make_tuple(x(0, 0), x(1, 0), x(2, 0), x(3, 0), std::get<4>(testSet[i]));
	}

	// create the model
	GLM::LogisticRegression<> logisticModel(4, nrOutputs);

	logisticModel.solver.alpha = 0.01;
	logisticModel.solver.beta1 = 0.7;
	logisticModel.solver.beta2 = 0.8;
	logisticModel.solver.lim = 1;

	Initializers::WeightsInitializerZero initializer;
	logisticModel.Initialize(initializer);

	// train the model

	const int batchSize = 64;

	Eigen::MatrixXd in(4, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, nrTraining - 1);
	for (int i = 0; i <= 1000; ++i)
	{
		for (int b = 0; b < batchSize; ++b)
		{
			const int ind = distIntBig(rde);
			const auto& record = trainingSet[ind];

			in(0, b) = std::get<0>(record);
			in(1, b) = std::get<1>(record);
			in(2, b) = std::get<2>(record);
			in(3, b) = std::get<3>(record);
			
			out(0, b) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
			if (nrOutputs > 1) out(1, b) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
			if (nrOutputs > 2) out(2, b) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;
		}
		logisticModel.AddBatch(in, out);
		if (i % 100 == 0)
		{
			double loss = logisticModel.getLoss() / batchSize;
			std::cout << "Loss: " << loss << std::endl;
		}
	}


	Utils::TestStatistics setosaStats;
	Utils::TestStatistics versicolorStats;
	Utils::TestStatistics virginicaStats;


	// test the model

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	for (const auto& record : trainingSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = logisticModel.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	std::cout << std::endl << "Setosa true positives: " << setosaStats.getTruePositives() << ", true negatives: " << setosaStats.getTrueNegatives() << ", false positives: " << setosaStats.getFalsePositives() << ", false negatives: " << setosaStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor true positives: " << versicolorStats.getTruePositives() << ", true negatives: " << versicolorStats.getTrueNegatives() << ", false positives: " << versicolorStats.getFalsePositives() << ", false negatives: " << versicolorStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica true positives: " << virginicaStats.getTruePositives() << ", true negatives: " << virginicaStats.getTrueNegatives() << ", false positives: " << virginicaStats.getFalsePositives() << ", false negatives: " << virginicaStats.getFalseNegatives() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa accuracy: " << setosaStats.getAccuracy() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor accuracy: " << versicolorStats.getAccuracy() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica accuracy: " << virginicaStats.getAccuracy() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa specificity: " << setosaStats.getSpecificity() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor specificity: " << versicolorStats.getSpecificity() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica specificity: " << virginicaStats.getSpecificity() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa precision: " << setosaStats.getPrecision() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor precision: " << versicolorStats.getPrecision() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica precision: " << virginicaStats.getPrecision() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa recall: " << setosaStats.getRecall() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor recall: " << versicolorStats.getRecall() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica recall: " << virginicaStats.getRecall() << std::endl;
	std::cout << std::endl;

	setosaStats.Clear();
	versicolorStats.Clear();
	virginicaStats.Clear();

	std::cout << std::endl << "Test set:" << std::endl;

	for (const auto& record : testSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;
		
		Eigen::VectorXd res = logisticModel.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	std::cout << std::endl << "Setosa true positives: " << setosaStats.getTruePositives() << ", true negatives: " << setosaStats.getTrueNegatives() << ", false positives: " << setosaStats.getFalsePositives() << ", false negatives: " << setosaStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor true positives: " << versicolorStats.getTruePositives() << ", true negatives: " << versicolorStats.getTrueNegatives() << ", false positives: " << versicolorStats.getFalsePositives() << ", false negatives: " << versicolorStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica true positives: " << virginicaStats.getTruePositives() << ", true negatives: " << virginicaStats.getTrueNegatives() << ", false positives: " << virginicaStats.getFalsePositives() << ", false negatives: " << virginicaStats.getFalseNegatives() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa accuracy: " << setosaStats.getAccuracy() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor accuracy: " << versicolorStats.getAccuracy() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica accuracy: " << virginicaStats.getAccuracy() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa specificity: " << setosaStats.getSpecificity() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor specificity: " << versicolorStats.getSpecificity() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica specificity: " << virginicaStats.getSpecificity() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa precision: " << setosaStats.getPrecision() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor precision: " << versicolorStats.getPrecision() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica precision: " << virginicaStats.getPrecision() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa recall: " << setosaStats.getRecall() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor recall: " << versicolorStats.getRecall() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica recall: " << virginicaStats.getRecall() << std::endl;
	std::cout << std::endl;

	return true;
}

bool LogisticRegressionTests()
{
	return SimpleLogisticRegressionTest() && MoreComplexLogisticRegressionTest() && IrisLogisticRegressionTest();
}