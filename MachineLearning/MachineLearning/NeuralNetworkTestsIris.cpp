#include "Tests.h"
#include "WeightsInitializer.h"
#include "NeuralNetwork.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Softmax.h"

void ShuffleIris(std::vector<Utils::IrisDataset::Record>& records, int nrTraining)
{
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

		if (setosa > 8 && versicolor > 8 && virginica > 8) break;
	}
}

void NormalizeIris(std::vector<Utils::IrisDataset::Record>& trainingSet, std::vector<Utils::IrisDataset::Record>& testSet, int nrOutputs = 3)
{
	// normalize the inputs
	const int nrTraining = static_cast<int>(trainingSet.size());

	Norm::InputOutputNormalizer normalizer(4, nrOutputs);
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
}


bool IrisNeuralNetworkTest()
{
	std::cout << std::endl << "Neural Network test for the Iris dataset, Setosa is lineary separable from the other two, but the others two cannot be linearly separated" << std::endl << std::endl;

	Utils::IrisDataset irisDataset;
	irisDataset.setRelativePath("../../Datasets/");
	irisDataset.setDataFileName("iris.data");

	if (!irisDataset.Open()) return false;

	auto records = irisDataset.getAllRecords();
	const int nrTraining = 120;

	// shuffle the data

	ShuffleIris(records, nrTraining);


	// split the data into training and test sets

	std::vector<Utils::IrisDataset::Record> trainingSet(records.begin(), records.begin() + nrTraining);
	std::vector<Utils::IrisDataset::Record> testSet(records.begin() + nrTraining, records.end());


	const int nrOutputs = 3; // 1 only for Setosa, 3 if all three classes are to be predicted

	// normalize the inputs
	NormalizeIris(trainingSet, testSet, nrOutputs);

	// create the model

	// more layers can be added, and/or made wider, but it will take more time to train. One configuration that I tried: { 4, 246, 512, 127, 63, 27, 9, nrOutputs }
	NeuralNetworks::MultilayerPerceptron<SGD::SoftmaxRegressionAdamSolver> neuralNetwork({ 4, 27, 9, nrOutputs });

	const double alpha = 0.01;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 1;

	neuralNetwork.setParams({ alpha, lim, beta1, beta2 });

	Initializers::WeightsInitializerXavierUniform initializer;
	neuralNetwork.Initialize(initializer);


	// train the model

	const int batchSize = 64;

	Eigen::MatrixXd in(4, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, nrTraining - 1);
	for (int i = 0; i <= 3000; ++i)
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
		neuralNetwork.ForwardBackwardStep(in, out);
		if (i % 300 == 0)
		{
			const double loss = neuralNetwork.getLoss() / batchSize;
			std::cout << "Loss: " << loss << std::endl;
		}
	}

	// test the model

	std::cout << std::endl << "Training set:" << std::endl;

	Utils::IrisDataset::PrintStats(trainingSet, nrOutputs, neuralNetwork);

	std::cout << std::endl << "Test set:" << std::endl;

	Utils::IrisDataset::PrintStats(testSet, nrOutputs, neuralNetwork);

	return true;
}


