#include "Tests.h"
#include "WeightsInitializer.h"
#include "NeuralNetwork.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Softmax.h"


bool XORNeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);
	
	const int nrTests = 1;

	// this alleviates the convergence issue
	// there are 16 local minima for xor where the 3 neurons network could get 'stuck'
	// ocassionally it might reach one but from my tests it can sometimes get out of it
	Initializers::WeightsInitializerForXorNetwork weightsInitializer;

	const double alpha = 0.01;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 0.1;

	// increasing the number of neurons from the minimum of 2 will help with the convergence
	// because the dimension of space is increased, there are paths out of the former local minimums that exist in the small dimensional space
	// using the special weights initializer is no longer necessary, but it shouldn't hurt either, in fact it seems to still help with the convergence
	// different parameters might be needed for an efficient convergence but I did not play with those yet
	const int numHiddenNeurons = 2;

	const int batchSize = 4;

	std::cout << std::endl << "XOR with neural network implemented 'in place' out of generalized linear models (last layer neuron being a logistic regression)" << std::endl << std::endl;

	Eigen::MatrixXd t(numHiddenNeurons, batchSize); // bogus target for hidden layer, won't be used except for its size
	Eigen::MatrixXd x, y;
	Eigen::VectorXd in(2);

	// try a simple neural network to solve the xor:
	int failures = 0;
	for (int trial = 0; trial < nrTests; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		// RMSProp or momentum also work

		// works with some other last neuron, such as one that has a tanh activation function, but I like the logistic one more
		//typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::TanhFunction<>> LastLayerRegressionAdamSolver;
		//GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, LastLayerRegressionAdamSolver> modelLastLayer(2, 1);

		GLM::LogisticRegression<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, SGD::LogisticRegressionAdamSolver> modelLastLayer(numHiddenNeurons, 1);

		modelLastLayer.getSolver().alpha = alpha;
		//modelLastLayer.getSolver().beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		modelLastLayer.getSolver().beta1 = beta1;
		modelLastLayer.getSolver().beta2 = beta2;
		modelLastLayer.getSolver().lim = lim;

		modelLastLayer.getSolver().firstLayer = false;

		// kind of works with tanh as well, it just seems to have a bigger chance to end up in a local minimum
		// works with others, too, but they might need some other parameters (for example, smaller aplha)
		//typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GLM::GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, numHiddenNeurons);

		hiddenLayerModel.getSolver().alpha = alpha;
		//hiddenLayerModel.getSolver().beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		hiddenLayerModel.getSolver().beta1 = beta1;
		hiddenLayerModel.getSolver().beta2 = beta2;
		hiddenLayerModel.getSolver().lim = lim;

		hiddenLayerModel.getSolver().lastLayer = false;

		hiddenLayerModel.Initialize(weightsInitializer);

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		int lowLoss = 0;
		for (int i = 0; i <= 10000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int x1 = distBool(rde);
				const int x2 = distBool(rde);
				x(0, b) = x1;
				x(1, b) = x2;

				y(0, b) = (x1 ^ x2);
			}

			// forward propagation
			// first layer:

			hiddenLayerModel.AddBatchNoParamsAdjustment(x, t);
			Eigen::MatrixXd pred = hiddenLayerModel.getPrediction();

			// forward and backward for the last layer:
			const Eigen::MatrixXd grad = modelLastLayer.AddBatch(pred, y);

			// now backpropagate the gradient to previous layer:
			pred = modelLastLayer.BackpropagateBatch(grad);

			// now do the adjustments as well in the first layer
			hiddenLayerModel.AddBatch(x, pred);

			if (i % 10000 == 0)
			{
				const double loss = modelLastLayer.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;

				if (loss < 1E-2)
					++lowLoss;
				else
					lowLoss = 0;

				if (lowLoss > 5) break;
			}
		}

		if (lowLoss <= 5)
			std::cout << "Failure to converge!" << std::endl;

		in(0) = 0;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		double res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 0 0 = " << res << std::endl;

		if (lowLoss > 5 && res >= 0.5)
			lowLoss = 0;

		in(0) = 0;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 0 1 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 1 0 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 1 1 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		//if (lowLoss < 5) return false;

		if (lowLoss <= 5) ++failures;
	}

	std::cout << std::endl << "Failures: " << failures << std::endl;

	std::cout << std::endl << std::endl << "XOR with the multilayer perceptron implementation" << std::endl << std::endl;

	const int failures_first = failures;

	
	failures = 0;
	x.resize(2, batchSize);
	y.resize(1, batchSize);

	for (int trial = 0; trial < nrTests; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		// with more neurons and even more layers it still works, for example { 2, 7, 5, 1 }, for some complex setup the initialization of weights should probably left to default
		NeuralNetworks::MultilayerPerceptron<> neuralNetwork({2, numHiddenNeurons, 1});

		neuralNetwork.setParams({alpha, lim, beta1, beta2});
		neuralNetwork.InitializHiddenLayers(weightsInitializer);

		int lowLoss = 0;
		for (int i = 0; i <= 10000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int x1 = distBool(rde);
				const int x2 = distBool(rde);
				x(0, b) = x1;
				x(1, b) = x2;

				y(0, b) = (x1 ^ x2);
			}

			neuralNetwork.ForwardBackwardStep(x, y);
			
			if (i % 10000 == 0)
			{
				const double loss = neuralNetwork.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;

				if (loss < 1E-2)
					++lowLoss;
				else
					lowLoss = 0;

				if (lowLoss > 5) break;
			}
		}

		if (lowLoss <= 5)
			std::cout << "Failure to converge!" << std::endl;

		in(0) = 0;
		in(1) = 0;
		double res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 0 0 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		in(0) = 0;
		in(1) = 1;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 0 1 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 0;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 1 0 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 1;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 1 1 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		if (lowLoss <= 5) ++failures;
	}
	
	std::cout << std::endl << "Failures: " << failures << std::endl;

	return failures_first == 0 && failures == 0;
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


	//for (auto rec : records)
	//	std::cout << std::get<0>(rec) << ", " << std::get<1>(rec) << ", " << std::get<2>(rec) << ", " << std::get<3>(rec) << ", " << std::get<4>(rec) << std::endl;

	// split the data into training and test sets
	
	std::vector<Utils::IrisDataset::Record> trainingSet(records.begin(), records.begin() + nrTraining);
	std::vector<Utils::IrisDataset::Record> testSet(records.begin() + nrTraining, records.end());

	// create the model
	const int nrOutputs = 3; // 1 only for Setosa, 3 if all three classes are to be predicted

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

	// more layers can be added, and/or made wider, but it will take more time to train. One configuration that I tried: { 4, 246, 512, 127, 63, 27, 9, nrOutputs }
	NeuralNetworks::MultilayerPerceptron<SGD::SoftmaxRegressionAdamSolver> neuralNetwork({4, 27, 9, nrOutputs});

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

			in(0, b) = std::get<0>(record);
			in(1, b) = std::get<1>(record);
			in(2, b) = std::get<2>(record);
			in(3, b) = std::get<3>(record);

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


	Utils::TestStatistics setosaStats;
	Utils::TestStatistics versicolorStats;
	Utils::TestStatistics virginicaStats;


	// test the model

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	long long int correct = 0;
	for (const auto& record : trainingSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = neuralNetwork.Predict(in.col(0));


		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);

		double limp = 0.5;
		for (int j = 0; j < nrOutputs; ++j)
			limp = std::max(limp, res(j));

		if (res(0) == limp && out(0, 0) > 0.5) ++correct;
		else if (nrOutputs > 1 && res(1) == limp && out(1, 0) > 0.5) ++correct;
		else if (nrOutputs > 2 && res(2) == limp && out(2, 0) > 0.5) ++correct;
	}

	Utils::TestStatistics totalStats;

	setosaStats.PrintStatistics("Setosa");
	if (nrOutputs > 1) {
		versicolorStats.PrintStatistics("Versicolor");
		if (nrOutputs > 2) virginicaStats.PrintStatistics("Virginica");

		totalStats.Add(setosaStats);
		totalStats.Add(versicolorStats);
		if (nrOutputs > 2) totalStats.Add(virginicaStats);

		//totalStats.PrintStatistics("Overall"); //misleading
	}


	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(trainingSet.size()) << "%" << std::endl << std::endl;

	setosaStats.Clear();
	versicolorStats.Clear();
	virginicaStats.Clear();

	std::cout << std::endl << "Test set:" << std::endl;

	correct = 0;

	for (const auto& record : testSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = neuralNetwork.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);

		double limp = 0.5;
		for (int j = 0; j < nrOutputs; ++j)
			limp = std::max(limp, res(j));

		if (res(0) == limp && out(0, 0) > 0.5) ++correct;
		else if (nrOutputs > 1 && res(1) == limp && out(1, 0) > 0.5) ++correct;
		else if (nrOutputs > 2 && res(2) == limp && out(2, 0) > 0.5) ++correct;
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

		//totalStats.PrintStatistics("Overall"); //misleading
	}

	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(testSet.size()) << "%" << std::endl << std::endl;

	return true;
}


bool NeuralNetworkTestsMNIST()
{
	std::cout << "MNIST Neural Network Tests, it will take a long time..." << std::endl;

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
	minstTestDataFiles.setImagesFileName("emnist-digits-test-images-idx3-ubyte");
	minstTestDataFiles.setLabelsFileName("emnist-digits-test-labels-idx1-ubyte");
	if (!minstTestDataFiles.Open()) {
		std::cout << "Couldn't load test data" << std::endl;
		return false;
	}

	std::vector<std::pair<std::vector<double>, uint8_t>> testRecords = minstTestDataFiles.ReadAllImagesAndLabels();
	minstTestDataFiles.Close();


	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(trainingRecords.begin(), trainingRecords.end(), g);

	// split the training data into training and validation sets

	const int nrTrainingRecords = static_cast<int>(trainingRecords.size() * 0.95);

	std::vector<std::pair<std::vector<double>, uint8_t>> validationRecords(trainingRecords.begin() + nrTrainingRecords, trainingRecords.end());
	trainingRecords.resize(nrTrainingRecords);

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


	Eigen::MatrixXd validationInputs(nrInputs, validationRecords.size());
	Eigen::MatrixXd validationOutputs(nrOutputs, validationRecords.size());
	Eigen::MatrixXd validationRes(nrOutputs, validationRecords.size());


	Eigen::MatrixXd trainStatsOutputs(nrOutputs, validationRecords.size());
	Eigen::MatrixXd trainStatsRes(nrOutputs, validationRecords.size());

	rec = 0;
	for (const auto& record : validationRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			validationInputs(i, rec) = record.first[i];
		for (int i = 0; i < nrOutputs; ++i)
			validationOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}


	Eigen::MatrixXd testInputs(nrInputs, testRecords.size());
	Eigen::MatrixXd testOutputs(nrOutputs, testRecords.size());


	rec = 0;
	for (const auto& record : testRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			testInputs(i, rec) = record.first[i];

		for (int i = 0; i < nrOutputs; ++i)
			testOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}

	// only inputs and only shifting the average

	trainInputs = trainInputs.colwise() - pixelsNormalizer.getAverageInput();
	validationInputs = validationInputs.colwise() - pixelsNormalizer.getAverageInput();
	testInputs = testInputs.colwise() - pixelsNormalizer.getAverageInput();

	// create the model
	// two hidden layers works quite well: { nrInputs, 1000, 100, nrOutputs } - use XavierUniform weights initializer for it - over 98%
	// also tested { nrInputs, 1000, 600, 100, nrOutputs } - use Glorot uniform weights initializer for it, this one I suspect that it needs different parameters and maybe more iterations
	// a single hidden layer, should be fast enough: { nrInputs, 32, nrOutputs } - over 97%
	// for simple ones the xavier initializer works well, for the deeper ones the glorot one is better
	NeuralNetworks::MultilayerPerceptron<SGD::SoftmaxRegressionAdamSolver> neuralNetwork(/*{nrInputs, 1000, 100, nrOutputs}*/ {nrInputs, 1000, 800, 400, 100, nrOutputs}, {0.2, 0.15, 0.1, 0, 0} ); // don't use dropout right before the softmax layer

#define LOAD_MODEL 1
#ifdef LOAD_MODEL
	// load some saved model

	if (!neuralNetwork.loadNetwork("../../data/neural45.net"))
	{
		std::cout << "Couldn't load the model" << std::endl;
		return false;
	}

#else
	// initialize the model
	double alpha = 0.0015; // non const, so it can be adjusted
	double decay = 0.95;
	const double beta1 = 0.9;
	const double beta2 = 0.95;
	const double lim = 10;

	neuralNetwork.setParams({ alpha, lim, beta1, beta2 });

	int startEpoch = 0; // set it to something different than 0 if you want to continue training


	if (startEpoch == 0)
	{
		//Initializers::WeightsInitializerXavierUniform initializer;
		Initializers::WeightsInitializerGlorotUniform initializer;
		//Initializers::WeightsInitializerHeNormal initializer;
		neuralNetwork.Initialize(initializer);
	}
	else
		// load some saved model
		if (!neuralNetwork.loadNetwork("../../data/neural" + std::to_string(startEpoch - 1) + ".net"))
		{
			std::cout << "Couldn't load the model" << std::endl;
			return false;
		}

	// train the model

	const int batchSize = 32;

	Eigen::MatrixXd in(nrInputs, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, static_cast<int>(trainInputs.cols() - 1));

// use dropout for input level instead!
//#define ADD_NOISE 1
#ifdef ADD_NOISE
	const double dropProb = 0.2; // also a hyperparameter
	std::bernoulli_distribution dist(dropProb);
#endif

	std::cout << "Training samples: " << trainInputs.cols() << std::endl;
	const long long int nrBatches	= trainInputs.cols() / batchSize;
	std::cout << "Traing batches / epoch: " << nrBatches << std::endl;

	std::cout << "Validation samples: " << validationInputs.cols() << std::endl;
	std::cout << "Test samples: " << testInputs.cols() << std::endl;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	const int nrEpochs = 60;
	
	std::vector<double> trainLosses(nrEpochs);
	std::vector<double> validationLosses(nrEpochs);

	std::vector<double> indices(nrEpochs);

	long long int bcnt = 0;
	for (int epoch = startEpoch; epoch < startEpoch + nrEpochs; ++epoch)
	{
		std::cout << "Epoch: " << epoch << " Alpha: " << alpha << std::endl;

		double totalLoss = 0;
		for (int batch = 0; batch < nrBatches; ++batch)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int ind = distIntBig(rde);

				in.col(b) = trainInputs.col(ind);

#ifdef ADD_NOISE
				for (int i = 0; i < nrInputs; ++i)
				{
					if (distDrop(rde))
						in(i, b) = 0;
				}
#endif

				out.col(b) = trainOutputs.col(ind);
			}

			neuralNetwork.ForwardBackwardStep(in, out);

			const double loss = neuralNetwork.getLoss() / batchSize;
			totalLoss += loss;
			
			if (bcnt % 100 == 0)
				std::cout << "Loss: " << loss << std::endl;
			
			++bcnt;
		}

		std::cout << "Average loss: " << totalLoss / static_cast<double>(nrBatches) << std::endl;


		// stats / epoch

		long long int validCorrect = 0;
		long long int trainCorrect = 0;

		for (int i = 0; i < validationRecords.size(); ++i)
		{
			Eigen::VectorXd res = neuralNetwork.Predict(validationInputs.col(i));
			validationRes.col(i) = res;

			double limp = 0;
			for (int j = 0; j < nrOutputs; ++j)
				limp = std::max(limp, res(j));

			int nr = -1;
			for (int j = 0; j < nrOutputs; ++j)
				if (validationOutputs(j, i) > 0.5)
				{
					if (nr != -1)
						std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
					nr = j;
				}

			int predn = -1;
			for (int n = 0; n < nrOutputs; ++n)
				if (res(n) >= limp)
				{
					if (predn != -1)
						std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
					predn = n;
				}

			if (predn == nr)
				++validCorrect;


			const int ind = distIntBig(rde);

			res = neuralNetwork.Predict(trainInputs.col(ind));
			trainStatsRes.col(i) = res;
			trainStatsOutputs.col(i) = trainOutputs.col(ind);

			limp = 0;

			for (int j = 0; j < nrOutputs; ++j)
				limp = std::max(limp, res(j));

			nr = -1;
			for (int j = 0; j < nrOutputs; ++j)
				if (trainStatsOutputs(j, i) > 0.5)
				{
					if (nr != -1)
						std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
					nr = j;
				}

			predn = -1;
			for (int n = 0; n < nrOutputs; ++n)
				if (res(n) >= limp)
				{
					if (predn != -1)
						std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
					predn = n;
				}

			if (predn == nr)
				++trainCorrect;
		}

		trainLosses[epoch] = neuralNetwork.getLoss(trainStatsRes, trainStatsOutputs) / static_cast<double>(validationRecords.size());
		validationLosses[epoch] = neuralNetwork.getLoss(validationRes, validationOutputs) / static_cast<double>(validationRecords.size());
		indices[epoch] = epoch;

		std::cout << "Training loss: " << trainLosses[epoch] << std::endl;
		std::cout << "Validation loss: " << validationLosses[epoch] << std::endl;

		std::cout << "Training accuracy: " << 100. * static_cast<double>(trainCorrect) / static_cast<double>(validationRecords.size()) << "%" << std::endl;
		std::cout << "Validation accuracy: " << 100. * static_cast<double>(validCorrect) / static_cast<double>(validationRecords.size()) << "%" << std::endl << std::endl;

		const std::string fileName = "../../data/neural" + std::to_string(epoch) + ".net";
		neuralNetwork.saveNetwork(fileName);

		// makes the learning rate smaller each epoch
		alpha *= decay;
		neuralNetwork.setLearnRate(alpha);
	}

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	std::cout << "Training took: " << dif / 1000. << " seconds!" << std::endl;

	{
		Utils::DataFileWriter theFile("../../data/EMNIST.txt");
		theFile.AddDataset(indices, trainLosses);
		theFile.AddDataset(indices, validationLosses);
	}

	Utils::Gnuplot plot;
	plot.setType(Utils::Gnuplot::ChartType::training);
	plot.setCmdFileName("EMNIST.plt");
	plot.setDataFileName("EMNIST.txt");
	plot.Execute();

#endif


	std::vector<Utils::TestStatistics> stats(nrOutputs);

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;
	
	long long int correct = 0;

	for (int i = 0; i < trainInputs.cols(); ++i)
	{
		Eigen::VectorXd res = neuralNetwork.Predict(trainInputs.col(i));

		double limp = 0;
		for (int j = 0; j < nrOutputs; ++j)
			limp = std::max(limp, res(j));

		int nr = -1;
		for (int j = 0; j < nrOutputs; ++j)
		{
			stats[j].AddPrediction(res(j) >= limp, trainOutputs(j, i) > 0.5);

			if (trainOutputs(j, i) > 0.5)
			{
				if (nr != -1)
					std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
				nr = j;
			}
		}

		int predn = -1;
		for (int n = 0; n < nrOutputs; ++n)
			if (res(n) >= limp)
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

	Utils::TestStatistics totalStats;
	for (int j = 0; j < nrOutputs; ++j)
		totalStats.Add(stats[j]);

	//totalStats.PrintStatistics("Overall"); //misleading

	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(trainInputs.cols()) << "%" << std::endl;

	// now, on test set:

	std::cout << std::endl << "Test set:" << std::endl;

	for (int j = 0; j < nrOutputs; ++j)
		stats[j].Clear();

	correct = 0;
	for (int i = 0; i < testInputs.cols(); ++i)
	{
		Eigen::VectorXd res = neuralNetwork.Predict(testInputs.col(i));

		double limp = 0;
		for (int j = 0; j < nrOutputs; ++j)
			limp = std::max(limp, res(j));

		int nr = -1;
		for (int j = 0; j < nrOutputs; ++j)
		{
			stats[j].AddPrediction(res(j) >= limp, testOutputs(j, i) > 0.5);

			if (testOutputs(j, i) > 0.5)
			{
				if (nr != -1)
					std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
				nr = j;
			}
		}

		int predn = -1;
		for (int n = 0; n < nrOutputs; ++n)
			if (res(n) >= limp)
			{
				if (predn != -1)
					std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
				predn = n;
			}

		if (predn == nr)
			++correct;

		if (i % 1000 == 0)
			std::cout << "Number: " << nr << " Prediction: " << predn << ((nr == predn) ? " Correct!" : " Wrong!") << std::endl;
	}

	for (int j = 0; j < nrOutputs; ++j)
		stats[j].PrintStatistics(std::to_string(j));

	totalStats.Clear();
	for (int j = 0; j < nrOutputs; ++j)
		totalStats.Add(stats[j]);

	//totalStats.PrintStatistics("Overall"); //misleading

	std::cout << "Accuracy (% correct): " << 100.0 * static_cast<double>(correct) / static_cast<double>(testInputs.cols()) << "%" << std::endl;

	return true;
}


bool NeuralNetworksTests()
{
	return XORNeuralNetworksTests() && IrisNeuralNetworkTest() && NeuralNetworkTestsMNIST();
}