#include "Tests.h"
#include "CSVDataFile.h"
#include "MNISTDatabase.h"
#include "Normalizer.h"

#include "Softmax.h"


void TrainModel(GLM::SoftmaxRegression<>& softmaxModel, int nrOutputs, int nrTraining, const std::vector<Utils::IrisDataset::Record>& trainingSet)
{
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

			Utils::IrisDataset::Get(record, in, b);

			out(0, b) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
			if (nrOutputs > 1) out(1, b) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
			if (nrOutputs > 2) out(2, b) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;
		}
		softmaxModel.AddBatch(in, out);
		if (i % 100 == 0)
		{
			const double loss = softmaxModel.getLoss() / batchSize;
			std::cout << "Loss: " << loss << std::endl;
		}
	}
}

bool SoftmaxTestsIris()
{
	std::cout << std::endl << "Softmax for the Iris dataset, Setosa is lineary separable from the other two, but the others two cannot be linearly separated, so expect good results for Setosa but not for the other two" << std::endl << std::endl;

	Utils::IrisDataset irisDataset;
	irisDataset.setRelativePath("../../Datasets/");
	irisDataset.setDataFileName("iris.data");

	if (!irisDataset.Open()) return false;

	auto records = irisDataset.getAllRecords();

	const int nrTraining = 100;

	Shuffle(records, nrTraining);

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
		Utils::IrisDataset::Get(trainingSet, x, i, i);

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
		Utils::IrisDataset::Get(testSet, x, i, 0);

		x.col(0) -= avgi;
		x.col(0) = x.col(0).cwiseProduct(istdi);

		testSet[i] = std::make_tuple(x(0, 0), x(1, 0), x(2, 0), x(3, 0), std::get<4>(testSet[i]));
	}

	// create the model
	GLM::SoftmaxRegression<> softmaxModel(4, nrOutputs);

	TrainModel(softmaxModel, nrOutputs, nrTraining, trainingSet);

	// test the model

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	Utils::IrisDataset::PrintStats(trainingSet, nrOutputs, softmaxModel);

	std::cout << std::endl << "Test set:" << std::endl;

	Utils::IrisDataset::PrintStats(testSet, nrOutputs, softmaxModel);

	return true;
}


bool SoftmaxTestsMNIST()
{
	std::cout << "MNIST Softmax Regression Tests" << std::endl;

	const int nrInputs = 28 * 28;
	const int nrOutputs = 10;

	std::vector<std::pair<std::vector<double>, uint8_t>> trainingRecords;
	std::vector<std::pair<std::vector<double>, uint8_t>> testRecords;

	if (!LoadData(trainingRecords, testRecords))
		return false;

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(trainingRecords.begin(), trainingRecords.end(), g);

	// normalize the data
	Norm::Normalizer<> pixelsNormalizer(nrInputs, nrOutputs);

	Eigen::MatrixXd trainInputs(nrInputs, trainingRecords.size());
	Eigen::MatrixXd trainOutputs(nrOutputs, trainingRecords.size());

	SetDataIntoMatrices(trainingRecords, trainInputs, trainOutputs);

	pixelsNormalizer.AddBatch(trainInputs, trainOutputs);


	Eigen::MatrixXd testInputs(nrInputs, testRecords.size());
	Eigen::MatrixXd testOutputs(nrOutputs, testRecords.size());

	SetDataIntoMatrices(testRecords, testInputs, testOutputs);

	// only inputs and only shifting the average

	trainInputs = trainInputs.colwise() - pixelsNormalizer.getAverageInput();
	testInputs = testInputs.colwise() - pixelsNormalizer.getAverageInput();

	// create the model
	GLM::SoftmaxRegression<> softmaxModel(nrInputs, nrOutputs);

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

		const double loss = softmaxModel.getLoss() / batchSize;
		std::cout << "Loss: " << loss << std::endl;
	}

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	Utils::MNISTDatabase::PrintStats(softmaxModel, trainInputs, trainOutputs, nrOutputs);

	// now, on test set:

	std::cout << std::endl << "Test set:" << std::endl;

	Utils::MNISTDatabase::PrintStats(softmaxModel, testInputs, testOutputs, nrOutputs);

	return true;
}

bool SoftmaxTests()
{
	return SoftmaxTestsIris() && SoftmaxTestsMNIST();
}