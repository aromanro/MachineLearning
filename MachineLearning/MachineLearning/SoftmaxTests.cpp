#include "Tests.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Normalizer.h"

#include "Softmax.h"


void getInVals(Eigen::VectorXd& in, const Utils::IrisDataset::Record& record)
{
	in(0) = std::get<0>(record);
	in(1) = std::get<1>(record);
	in(2) = std::get<2>(record);
	in(3) = std::get<3>(record);
}

double getMax(const Eigen::VectorXd& res,int nrOutputs)
{
	double limp = 0.5;
	for (int j = 0; j < nrOutputs; ++j)
		limp = std::max(limp, res(j));

	return limp;
}

void CountCorrect(const Eigen::VectorXd& res, const Eigen::VectorXd& out, int nrOutputs, long long int& correct)
{
	const double limp = getMax(res, nrOutputs);

	if (res(0) == limp && out(0) > 0.5) ++correct;
	else if (nrOutputs > 1 && res(1) == limp && out(1) > 0.5) ++correct;
	else if (nrOutputs > 2 && res(2) == limp && out(2) > 0.5) ++correct;
}

void PrintStats(const std::vector<Utils::IrisDataset::Record>& records, int nrOutputs, GLM::SoftmaxRegression<>& softmaxModel)
{
	Utils::TestStatistics setosaStats;
	Utils::TestStatistics versicolorStats;
	Utils::TestStatistics virginicaStats;

	Eigen::VectorXd in(4);
	Eigen::VectorXd out(nrOutputs);

	long long int correct = 0;
	for (const auto& record : records)
	{
		getInVals(in, record);

		out(0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = softmaxModel.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0) > 0.5);

		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2) > 0.5);

		CountCorrect(res, out, nrOutputs, correct);
	}

	setosaStats.PrintStatistics("Setosa");
	if (nrOutputs > 1) {
		versicolorStats.PrintStatistics("Versicolor");
		if (nrOutputs > 2) virginicaStats.PrintStatistics("Virginica");
	}

	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(records.size()) << "%" << std::endl << std::endl;
}


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

	TrainModel(softmaxModel, nrOutputs, nrTraining, trainingSet);

	// test the model

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	PrintStats(trainingSet, nrOutputs, softmaxModel);

	std::cout << std::endl << "Test set:" << std::endl;

	PrintStats(testSet, nrOutputs, softmaxModel);

	return true;
}


void PrintStats(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& outputs, int nrOutputs, GLM::SoftmaxRegression<>& softmaxModel)
{
	std::vector<Utils::TestStatistics> stats(nrOutputs);

	// first, on training set:

	long long int correct = 0;

	for (int i = 0; i < inputs.cols(); ++i)
	{
		Eigen::VectorXd res = softmaxModel.Predict(inputs.col(i));

		double lim = 0;
		for (int j = 0; j < nrOutputs; ++j)
			lim = std::max(lim, res(j));

		int nr = -1;
		for (int j = 0; j < nrOutputs; ++j)
		{
			stats[j].AddPrediction(res(j) >= lim, outputs(j, i) > 0.5);

			if (outputs(j, i) > 0.5)
			{
				if (nr != -1)
					std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
				nr = j;
			}
		}

		int predn = -1;
		for (int n = 0; n < nrOutputs; ++n)
			if (res(n) >= lim)
			{
				if (predn != -1)
					std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
				predn = n;
			}

		if (predn == nr)
			++correct;
	}

	for (int j = 0; j < nrOutputs; ++j)
		stats[j].PrintStatistics(std::to_string(j));

	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(inputs.cols()) << "%" << std::endl;
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

	PrintStats(trainInputs, trainOutputs, nrOutputs, softmaxModel);

	// now, on test set:

	std::cout << std::endl << "Test set:" << std::endl;

	PrintStats(testInputs, testOutputs, nrOutputs, softmaxModel);

	return true;
}

bool SoftmaxTests()
{
	return SoftmaxTestsIris() && SoftmaxTestsMNIST();
}