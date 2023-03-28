#include "Tests.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Normalizer.h"

#include "Softmax.h"

bool SoftmaxTestsIris()
{
	std::cout << std::endl << "Softmax for the Iris dataset, Setosa is lineary separable from the other two, but the others two cannot be linearly separated, so expect good results for Setosa but not for the other two" << std::endl << std::endl;

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
			const auto& rec = *it;
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

	const int nrOutputs = 3;

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
	GLM::SoftmaxRegression<> softmaxModel(4, nrOutputs);

	softmaxModel.getSolver().alpha = 0.01;
	softmaxModel.getSolver().beta1 = 0.7;
	softmaxModel.getSolver().beta2 = 0.8;
	softmaxModel.getSolver().lim = 1;

	Initializers::WeightsInitializerZero initializer;
	softmaxModel.Initialize(initializer);

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
		softmaxModel.AddBatch(in, out);
		if (i % 100 == 0)
		{
			double loss = softmaxModel.getLoss() / batchSize;
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

		Eigen::VectorXd res = softmaxModel.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	Utils::TestStatistics totalStats;

	setosaStats.PrintStatistics("Setosa");
	if (nrOutputs > 1) {
		versicolorStats.PrintStatistics("Versicolor");
		if (nrOutputs > 2) virginicaStats.PrintStatistics("Virginica");
	
		totalStats.Add(setosaStats);
		totalStats.Add(versicolorStats);
		if (nrOutputs > 2) totalStats.Add(virginicaStats);

		totalStats.PrintStatistics("Overall");
	}

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

		Eigen::VectorXd res = softmaxModel.Predict(in.col(0));

		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	setosaStats.PrintStatistics("Setosa");
	if (nrOutputs > 1) 
	{
		versicolorStats.PrintStatistics("Versicolor");
		if (nrOutputs > 2) virginicaStats.PrintStatistics("Virginica");

		totalStats.Clear();
		totalStats.Add(setosaStats);
		totalStats.Add(versicolorStats);
		if (nrOutputs > 2) totalStats.Add(virginicaStats);

		totalStats.PrintStatistics("Overall");
	}

	std::cout << std::endl;

	return true;
}


bool SoftmaxTestsMNIST()
{
	std::cout << "MNIST Softmax Regression Tests" << std::endl;

	const int nrInputs = 28 * 28;
	const int nrOutputs = 10;

	// load the data
	Utils::MNISTDatabase minstTrainDataFiles;
	if (!minstTrainDataFiles.Open()) {
		std::cout << "Couldn't load train data" << std::endl;
		return false;
	}

	std::vector<std::pair<std::vector<double>, uint8_t>> trainingRecords = minstTrainDataFiles.ReadAllImagesAndLabels();
	minstTrainDataFiles.Close();

	Utils::MNISTDatabase minstTestDataFiles;
	minstTrainDataFiles.setImagesFileName("emnist-digits-test-images-idx3-ubyte");
	minstTrainDataFiles.setLabelsFileName("emnist-digits-test-labels-idx1-ubyte");
	if (!minstTestDataFiles.Open()) {
		std::cout << "Couldn't load test data" << std::endl;
		return false;
	}

	std::vector<std::pair<std::vector<double>, uint8_t>> testRecords = minstTestDataFiles.ReadAllImagesAndLabels();
	minstTestDataFiles.Close();


	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(trainingRecords.begin(), trainingRecords.end(), g);

	// normalize the data
	Norm::Normalizer<> pixelsNormalizer(nrInputs, nrOutputs);

	Eigen::MatrixXd trainInputs(nrInputs, trainingRecords.size());
	Eigen::MatrixXd trainOutputs(nrOutputs, trainingRecords.size());

	int rec = 0;
	for (const auto& record : trainingRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			trainInputs(i, rec) = record.first[i];

		for (int i = 0; i < nrOutputs; ++i)
			trainOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}

	pixelsNormalizer.AddBatch(trainInputs, trainOutputs);


	Eigen::MatrixXd testInputs(nrInputs, testRecords.size());
	Eigen::MatrixXd testOutputs(10, testRecords.size());

	rec = 0;
	for (const auto& record : testRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			testInputs(i, rec) = record.first[i];

		for (int i = 0; i < 10; ++i)
			testOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}

	// only inputs and only shifting the average

	trainInputs = trainInputs.colwise() - pixelsNormalizer.getAverageInput();
	testInputs = testInputs.colwise() - pixelsNormalizer.getAverageInput();

	// create the model
	GLM::SoftmaxRegression<> softmaxModel(nrInputs, 10);

	softmaxModel.getSolver().alpha = 0.0005;
	softmaxModel.getSolver().beta1 = 0.7;
	softmaxModel.getSolver().beta2 = 0.9;
	softmaxModel.getSolver().lim = 1;

	Initializers::WeightsInitializerZero initializer;
	softmaxModel.Initialize(initializer);

	// train the model

	const int batchSize = 128;

	Eigen::MatrixXd in(nrInputs, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, static_cast<int>(trainInputs.cols() - 1));
	for (int epoch = 0; epoch < 20; ++epoch)
	{
		for (int batch = 0; batch < trainInputs.cols() / batchSize; ++batch)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int ind = distIntBig(rde);

				in.col(b) = trainInputs.col(ind);
				out.col(b) = trainOutputs.col(ind);

			}

			softmaxModel.AddBatch(in, out);
		}

		double loss = softmaxModel.getLoss() / batchSize;
		std::cout << "Loss: " << loss << std::endl;
	}

	std::vector<Utils::TestStatistics> stats(10);

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	for (int i = 0; i < trainInputs.cols(); ++i)
	{
		Eigen::VectorXd res = softmaxModel.Predict(trainInputs.col(i));

		double lim = 0;
		for (int j = 0; j < 10; ++j)
			lim = max(lim, res(j));

		for (int j = 0; j < 10; ++j)
			stats[j].AddPrediction(res(j) >= lim, trainOutputs(j, i) > 0.5);
	}

	for (int j = 0; j < 10; ++j)
		stats[j].PrintStatistics(std::to_string(j));

	Utils::TestStatistics totalStats;
	for (int j = 0; j < 10; ++j)
		totalStats.Add(stats[j]);

	totalStats.PrintStatistics("Overall");

	// now, on test set:

	std::cout << std::endl << "Test set:" << std::endl;

	for (int j = 0; j < 10; ++j)
		stats[j].Clear();

	for (int i = 0; i < testInputs.cols(); ++i)
	{
		Eigen::VectorXd res = softmaxModel.Predict(testInputs.col(i));

		double lim = 0;
		for (int j = 0; j < 10; ++j)
			lim = max(lim, res(j));

		for (int j = 0; j < 10; ++j)
			stats[j].AddPrediction(res(j) >= lim, testOutputs(j, i) > 0.5);

		if (i % 1000 == 0)
		{
			int nr = -1;
			for (int n = 0; n < 10; ++n)
				if (testOutputs(n, i) > 0.5)
				{
					if (nr != -1)
						std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << n << std::endl;
					nr = n;
				}

			int predn = -1;
			for (int n = 0; n < 10; ++n)
				if (res(n) >= lim)
				{
					if (predn != -1)
						std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
					predn = n;
				}

			std::cout << "Number: " << nr << " Prediction: " << predn << ((nr == predn) ? " Correct!" : " Wrong!") << std::endl;
		}
	}

	for (int j = 0; j < 10; ++j)
		stats[j].PrintStatistics(std::to_string(j));

	totalStats.Clear();
	for (int j = 0; j < 10; ++j)
		totalStats.Add(stats[j]);

	totalStats.PrintStatistics("Overall");

	return true;
}

bool SoftmaxTests()
{
	return SoftmaxTestsIris() && SoftmaxTestsMNIST();
}